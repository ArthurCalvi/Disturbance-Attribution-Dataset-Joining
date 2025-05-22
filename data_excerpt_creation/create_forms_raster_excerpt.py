import os
import logging
import rasterio
from rasterio.mask import mask as rio_mask
# from rasterio.warp import transform_geom # Not directly used for BBOX transformation here, geopandas handles it
from shapely.geometry import box
import geopandas as gpd
import numpy as np
import math # For sqrt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Path to the single FORMS GeoTIFF file
FORMS_RASTER_FILE_PATH = '/Users/arthurcalvi/Data/Disturbances_maps/FORMS/Height_mavg_2023.tif'

# BBOX coordinates (minx, miny, maxx, maxy) in EPSG:2154
BBOX_COORDS_EPSG2154 = (307783.0822, 6340505.4366, 469246.8845, 6419190.9011)
BBOX_CRS_EPSG2154 = "EPSG:2154"
# The notebook suggests the input _l93.tif files are likely EPSG:2154 already,
# but it's good to have a fallback if CRS is missing or different.
# The notebook also mentions converting to EPSG:3035 for some steps, but for raw excerpt, we stick to input or target.
EXPECTED_RASTER_CRS = "EPSG:2154" # Assuming L93 implies EPSG:2154, verify with file if possible.

# Output directory relative to this script's location
OUTPUT_RASTER_DIR_RELATIVE = "../excerpts/raw/"
OUTPUT_RASTER_FILENAME_PREFIX = "excerpt_forms_height_mavg_2023"

# File size limits (MB)
HARD_MAX_SIZE_MB = 10.0
# Fallback attempts if direct scaling is slightly off
FALLBACK_SCALE_FACTOR = 0.95 # Scale area by this factor (sides by sqrt of this)
MAX_FALLBACK_ATTEMPTS = 2
SAFETY_MARGIN_FACTOR = 0.98 # Aim for slightly smaller than 10MB to be safe

# Data type optimization
# The notebook uses int16 for height data and later uint8 for difference maps.
# For the raw height map, int16 is likely appropriate if values represent cm.
# If the values are already scaled or represent something else, this might need adjustment.
OUTPUT_DTYPE = 'int16' # Assuming height in cm, adjust if needed after inspecting raw file.
DEFAULT_NODATA_VALUE_FOR_OUTPUT_DTYPE = np.iinfo(OUTPUT_DTYPE).min # A common practice for int types

def create_raster_excerpt_with_size_control(
    src_path,
    output_path,
    initial_bbox_geom_src_crs, # Initial BBOX geometry already in source CRS
    source_profile,
    source_nodata_val,
    target_dtype,
    target_nodata_val_for_output,
    max_size_mb,
    fallback_scale_factor = 0.95, # Scales area, so sides by sqrt(fallback_scale_factor)
    max_fallback_attempts = 2,
    safety_margin = 0.98 # Aim for slightly smaller than max_size_mb
):
    """Crops a raster to a given BBOX, attempting to meet file size constraints."""
    current_bbox_geom = initial_bbox_geom_src_crs
    final_cropped_image = None
    final_out_transform = None
    final_out_meta = None
    final_file_size_mb = float('inf')

    # Attempt 1: Direct scaling based on initial full BBOX crop (if needed)
    # First, let's try with the initial_bbox_geom_src_crs directly
    logging.info(f"Attempting crop with initial BBOX (in src_crs): {current_bbox_geom.bounds}")
    try:
        out_image_full_bbox, out_transform_full_bbox = rio_mask(
            rasterio.open(src_path), [current_bbox_geom], crop=True, all_touched=True, 
            nodata=source_nodata_val if source_nodata_val is not None else 0
        )
    except ValueError as ve:
        if "Input shapes do not overlap raster." in str(ve):
            logging.warning("Initial BBOX does not overlap raster. Output will be minimal.")
            out_image_full_bbox = np.full((source_profile['count'], 1, 1), 
                                         (source_nodata_val if source_nodata_val is not None else DEFAULT_NODATA_VALUE_FOR_OUTPUT_DTYPE), 
                                         dtype=source_profile['dtype'])
            pixel_size_x = source_profile['transform'].a
            pixel_size_y = source_profile['transform'].e
            crop_center_x = current_bbox_geom.centroid.x
            crop_center_y = current_bbox_geom.centroid.y
            out_transform_full_bbox = rasterio.Affine(pixel_size_x, 0.0, crop_center_x - pixel_size_x / 2,
                                                  0.0, pixel_size_y, crop_center_y - pixel_size_y / 2)
        else: raise ve

    out_meta_full_bbox = source_profile.copy()
    out_meta_full_bbox.update({
        "height": out_image_full_bbox.shape[1],
        "width": out_image_full_bbox.shape[2],
        "transform": out_transform_full_bbox,
        "dtype": target_dtype,
        "nodata": target_nodata_val_for_output,
        "compress": "lzw"
    })
    
    temp_output_path = output_path + "_temp.tif"
    
    # Prepare data for writing (dtype conversion and nodata handling)
    img_to_write_full_bbox = np.full(out_image_full_bbox.shape, target_nodata_val_for_output, dtype=target_dtype)
    if source_nodata_val is not None:
        valid_mask = (out_image_full_bbox != source_nodata_val)
        img_to_write_full_bbox[valid_mask] = out_image_full_bbox[valid_mask].astype(target_dtype)
    else:
        img_to_write_full_bbox = out_image_full_bbox.astype(target_dtype)

    with rasterio.open(temp_output_path, "w", **out_meta_full_bbox) as dest:
        dest.write(img_to_write_full_bbox)
    
    size_initial_crop_mb = os.path.getsize(temp_output_path) / (1024 * 1024)
    logging.info(f"Size after initial full BBOX crop: {size_initial_crop_mb:.2f} MB")

    if size_initial_crop_mb <= max_size_mb:
        logging.info("Initial crop is within size limits.")
        os.rename(temp_output_path, output_path)
        final_file_size_mb = size_initial_crop_mb
        current_bbox_for_log = current_bbox_geom # Save for final log
    else:
        logging.info("Initial crop exceeds size limit. Attempting direct scaling.")
        area_initial_bbox = current_bbox_geom.area
        # Target slightly less than max_size_mb for safety
        size_ratio_for_area = (max_size_mb * safety_margin) / size_initial_crop_mb 
        
        if size_ratio_for_area <= 0: # Safety check if somehow max_size_mb is tiny or initial is huge
            logging.warning("Calculated size ratio is zero or negative. Will use a very small fallback scale.")
            size_ratio_for_area = (fallback_scale_factor**max_fallback_attempts)**2 # Drastic reduction

        side_scale_factor = math.sqrt(size_ratio_for_area)

        bounds = current_bbox_geom.bounds
        minx, miny, maxx, maxy = bounds
        width = maxx - minx
        height = maxy - miny
        center_x = minx + width / 2
        center_y = miny + height / 2

        new_width = width * side_scale_factor
        new_height = height * side_scale_factor

        scaled_minx = center_x - new_width / 2
        scaled_maxx = center_x + new_width / 2
        scaled_miny = center_y - new_height / 2
        scaled_maxy = center_y + new_height / 2
        current_bbox_geom = box(scaled_minx, scaled_miny, scaled_maxx, scaled_maxy)
        logging.info(f"Calculated scaled BBOX (in src_crs): {current_bbox_geom.bounds} with side_scale_factor {side_scale_factor:.3f}")

        # Now loop with fallback attempts using this scaled BBOX as the starting point
        for attempt in range(max_fallback_attempts + 1): # +1 for the initial scaled attempt
            if attempt > 0:
                logging.info(f"Fallback attempt {attempt}/{max_fallback_attempts}.")
                # Scale area of current_bbox_geom by fallback_scale_factor
                # sides by sqrt(fallback_scale_factor)
                fb_side_scale = math.sqrt(fallback_scale_factor)
                bounds = current_bbox_geom.bounds
                minx, miny, maxx, maxy = bounds
                width = maxx - minx
                height = maxy - miny
                center_x = minx + width / 2
                center_y = miny + height / 2
                new_width = width * fb_side_scale
                new_height = height * fb_side_scale
                current_bbox_geom = box(center_x - new_width / 2, center_y - new_height / 2, 
                                        center_x + new_width / 2, center_y + new_height / 2)
                logging.info(f"Fallback scaled BBOX (in src_crs): {current_bbox_geom.bounds}")

            try:
                out_image, out_transform = rio_mask(
                    rasterio.open(src_path), [current_bbox_geom], crop=True, all_touched=True, 
                    nodata=source_nodata_val if source_nodata_val is not None else 0
                )
            except ValueError as ve: # Catch if BBOX doesn't overlap
                if "Input shapes do not overlap raster." in str(ve):
                    logging.warning(f"BBOX at attempt (scaled/fallback) does not overlap raster. Output will be minimal.")
                    out_image = np.full((source_profile['count'], 1, 1), 
                                         (source_nodata_val if source_nodata_val is not None else DEFAULT_NODATA_VALUE_FOR_OUTPUT_DTYPE), 
                                         dtype=source_profile['dtype'])
                    pixel_size_x = source_profile['transform'].a
                    pixel_size_y = source_profile['transform'].e
                    crop_center_x = current_bbox_geom.centroid.x
                    crop_center_y = current_bbox_geom.centroid.y
                    out_transform = rasterio.Affine(pixel_size_x, 0.0, crop_center_x - pixel_size_x / 2,
                                                  0.0, pixel_size_y, crop_center_y - pixel_size_y / 2)
                else: raise ve
            
            # Handle cases where crop results in 0-dimension image
            if out_image.shape[1] == 0 or out_image.shape[2] == 0:
                logging.warning(f"Cropped image has zero width/height. Output will be minimal. Shape: {out_image.shape}")
                h = 1 if out_image.shape[1] == 0 else out_image.shape[1]
                w = 1 if out_image.shape[2] == 0 else out_image.shape[2]
                out_image = np.full((source_profile['count'], h, w), 
                                     (source_nodata_val if source_nodata_val is not None else DEFAULT_NODATA_VALUE_FOR_OUTPUT_DTYPE), 
                                     dtype=source_profile['dtype'])
                if h == 1 and w == 1 and out_meta_full_bbox['transform']: # reuse initial transform if it was 1x1 too
                    out_transform = out_meta_full_bbox['transform']
                 # Else, out_transform from rio_mask might be problematic for 0-dim, this is a simplified recovery


            out_meta = source_profile.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "dtype": target_dtype,
                "nodata": target_nodata_val_for_output,
                "compress": "lzw"
            })

            # Prepare data for writing
            img_to_write = np.full(out_image.shape, target_nodata_val_for_output, dtype=target_dtype)
            if source_nodata_val is not None:
                valid_mask = (out_image != source_nodata_val)
                img_to_write[valid_mask] = out_image[valid_mask].astype(target_dtype)
            else:
                img_to_write = out_image.astype(target_dtype)

            with rasterio.open(temp_output_path, "w", **out_meta) as dest:
                dest.write(img_to_write)
            
            current_file_size_mb = os.path.getsize(temp_output_path) / (1024 * 1024)
            logging.info(f"Size after scaling/fallback attempt {attempt+1 if attempt >0 else ' (direct scale)'}: {current_file_size_mb:.2f} MB. BBOX ({source_profile['crs']}): {current_bbox_geom.bounds}")

            if current_file_size_mb <= max_size_mb:
                os.rename(temp_output_path, output_path)
                final_file_size_mb = current_file_size_mb
                current_bbox_for_log = current_bbox_geom
                logging.info(f"Success: File size {final_file_size_mb:.2f}MB is within limit.")
                break
            elif attempt == max_fallback_attempts:
                os.rename(temp_output_path, output_path) # Save the smallest one we got
                final_file_size_mb = current_file_size_mb
                current_bbox_for_log = current_bbox_geom
                logging.critical(f"Failed to reduce file below {max_size_mb}MB. Smallest achieved: {final_file_size_mb:.2f}MB.")
                break # Break from fallback loop
        else: # If loop finishes without break (only if initial scaled version was already too small and max_fallback_attempts = 0)
            if os.path.exists(temp_output_path):
                 os.rename(temp_output_path, output_path)
                 final_file_size_mb = current_file_size_mb
                 current_bbox_for_log = current_bbox_geom

    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)
    
    return final_file_size_mb, current_bbox_for_log if 'current_bbox_for_log' in locals() else initial_bbox_geom_src_crs

def create_forms_raster_excerpt():
    """Creates a clipped and compressed excerpt from the FORMS height raster, iteratively reducing BBOX if > HARD_MAX_SIZE_MB."""
    logging.info(f"Starting FORMS raster excerpt creation for {FORMS_RASTER_FILE_PATH}")

    # Create output directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_output_dir = os.path.join(script_dir, OUTPUT_RASTER_DIR_RELATIVE)
    os.makedirs(abs_output_dir, exist_ok=True)
    base_output_filename = f"{OUTPUT_RASTER_FILENAME_PREFIX}.tif"
    abs_output_path = os.path.join(abs_output_dir, base_output_filename)

    try:
        with rasterio.open(FORMS_RASTER_FILE_PATH) as src:
            source_profile = src.meta.copy()
            source_crs = src.crs
            source_dtype_str = src.dtypes[0] # dtype as string e.g. 'int16'
            source_nodata_val = src.nodatavals[0]
            logging.info(f"Source raster CRS: {source_crs}, Dtype: {source_dtype_str}, Nodata: {source_nodata_val}")

            if not source_crs:
                logging.warning(f"Source raster has no CRS. Assuming {EXPECTED_RASTER_CRS}.")
                source_crs = rasterio.CRS.from_string(EXPECTED_RASTER_CRS)
                source_profile['crs'] = source_crs # Update profile if CRS was missing

            initial_bbox_epsg2154 = box(*BBOX_COORDS_EPSG2154)
            bbox_gdf_epsg2154 = gpd.GeoDataFrame([{'geometry': initial_bbox_epsg2154}], crs=BBOX_CRS_EPSG2154)

            if source_crs != rasterio.CRS.from_string(BBOX_CRS_EPSG2154):
                logging.info(f"Reprojecting BBOX from {BBOX_CRS_EPSG2154} to {source_crs} for cropping.")
                initial_cropping_gdf_src_crs = bbox_gdf_epsg2154.to_crs(source_crs)
            else:
                initial_cropping_gdf_src_crs = bbox_gdf_epsg2154
            
            initial_cropping_geom_src_crs = initial_cropping_gdf_src_crs.geometry.iloc[0]

            # Determine target_nodata_val for OUTPUT_DTYPE
            target_nodata_val = DEFAULT_NODATA_VALUE_FOR_OUTPUT_DTYPE
            if source_nodata_val is not None:
                try:
                    casted_source_nodata = np.array([source_nodata_val]).astype(OUTPUT_DTYPE)[0]
                    target_type_info = np.iinfo(OUTPUT_DTYPE) if np.issubdtype(np.dtype(OUTPUT_DTYPE), np.integer) else np.finfo(OUTPUT_DTYPE)
                    if float(casted_source_nodata) == float(source_nodata_val) and \
                       target_type_info.min <= casted_source_nodata <= target_type_info.max:
                        target_nodata_val = casted_source_nodata
                        logging.info(f"Source nodata ({source_nodata_val}) can be represented as {target_nodata_val} in {OUTPUT_DTYPE}.")
                    else:
                        logging.warning(f"Source nodata ({source_nodata_val}) cannot be precisely represented or is out of range for {OUTPUT_DTYPE}. Using default {target_nodata_val}.")
                except (ValueError, OverflowError):
                    logging.warning(f"Source nodata ({source_nodata_val}) error on cast to {OUTPUT_DTYPE}. Using default {target_nodata_val}.")
            
            final_size, final_bbox = create_raster_excerpt_with_size_control(
                src_path=FORMS_RASTER_FILE_PATH,
                output_path=abs_output_path,
                initial_bbox_geom_src_crs=initial_cropping_geom_src_crs,
                source_profile=source_profile, # Pass the full source profile
                source_nodata_val=source_nodata_val,
                target_dtype=OUTPUT_DTYPE,
                target_nodata_val_for_output=target_nodata_val,
                max_size_mb=HARD_MAX_SIZE_MB,
                fallback_scale_factor=FALLBACK_SCALE_FACTOR,
                max_fallback_attempts=MAX_FALLBACK_ATTEMPTS,
                safety_margin=SAFETY_MARGIN_FACTOR
            )

            if final_size <= HARD_MAX_SIZE_MB:
                 logging.info(f"FORMS excerpt successfully created at {abs_output_path}. Final size: {final_size:.2f} MB. Used BBOX (in {source_crs}): {final_bbox.bounds}")
            else:
                 logging.critical(f"FORMS excerpt at {abs_output_path} is {final_size:.2f} MB, EXCEEDING {HARD_MAX_SIZE_MB} MB limit. Used BBOX (in {source_crs}): {final_bbox.bounds}")

    except FileNotFoundError:
        logging.error(f"ERROR: Raw FORMS raster file not found at {FORMS_RASTER_FILE_PATH}")
    except Exception as e:
        logging.error(f"An error occurred during FORMS raster excerpt creation: {e}", exc_info=True)

if __name__ == "__main__":
    create_forms_raster_excerpt() 