import os
import logging
import rasterio
from rasterio.mask import mask as rio_mask
# from rasterio.warp import transform_geom # Not directly used for BBOX transformation here, geopandas handles it
from shapely.geometry import box
import geopandas as gpd
import numpy as np
import math # For sqrt
import tempfile # Added for temporary directory
import shutil # Added for removing temporary directory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Path to the FORMS GeoTIFF files - now a list
FORMS_RASTER_FILE_PATHS = [
    '/Users/arthurcalvi/Data/Disturbances_maps/FORMS/Height_mavg_2023.tif',
    '/Users/arthurcalvi/Data/Disturbances_maps/FORMS/Height_mavg_2022.tif'
]

# BBOX coordinates (minx, miny, maxx, maxy) in EPSG:2154
BBOX_COORDS_EPSG2154 = (307783.0822, 6340505.4366, 469246.8845, 6419190.9011)
BBOX_CRS_EPSG2154 = "EPSG:2154"
# The notebook suggests the input _l93.tif files are likely EPSG:2154 already,
# but it's good to have a fallback if CRS is missing or different.
# The notebook also mentions converting to EPSG:3035 for some steps, but for raw excerpt, we stick to input or target.
EXPECTED_RASTER_CRS = "EPSG:2154" # Assuming L93 implies EPSG:2154, verify with file if possible.

# Output directory relative to this script's location
OUTPUT_RASTER_DIR_RELATIVE = "../excerpts/raw/"
# OUTPUT_RASTER_FILENAME_PREFIX = "excerpt_forms_height_mavg_2023" # Will be generated dynamically

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

# --- Helper Functions ---

def _get_raster_info(src_path):
    """Opens a raster and returns its profile, CRS, dtype string, and nodata value."""
    with rasterio.open(src_path) as src:
        profile = src.meta.copy()
        crs = src.crs
        dtype_str = src.dtypes[0]
        nodata_val = src.nodatavals[0]

        if not crs:
            logging.warning(f"Source raster {src_path} has no CRS. Assuming {EXPECTED_RASTER_CRS}.")
            crs = rasterio.CRS.from_string(EXPECTED_RASTER_CRS)
            profile['crs'] = crs
        return profile, crs, dtype_str, nodata_val

def _scale_bbox(bbox_geom, side_scale_factor):
    """Scales a Shapely geometry BBOX from its center."""
    if side_scale_factor == 1.0:
        return bbox_geom
    minx, miny, maxx, maxy = bbox_geom.bounds
    width = maxx - minx
    height = maxy - miny
    center_x = minx + width / 2
    center_y = miny + height / 2
    new_width = width * side_scale_factor
    new_height = height * side_scale_factor
    return box(center_x - new_width / 2, center_y - new_height / 2,
               center_x + new_width / 2, center_y + new_height / 2)

def _crop_and_write_raster(
    src_path,
    output_path,
    bbox_geom_src_crs, # BBOX geometry already in source CRS
    source_profile, # Full profile of the source raster
    source_nodata_val, # Nodata value from the source raster
    target_dtype,
    target_nodata_val_for_output
):
    """
    Crops a raster to a given BBOX and writes it to output_path.
    Returns the file size in MB.
    Handles non-overlapping BBOX by creating a minimal raster.
    """
    logging.debug(f"Cropping {src_path} to BBOX: {bbox_geom_src_crs.bounds} for output: {output_path}")
    try:
        out_image, out_transform = rio_mask(
            rasterio.open(src_path), [bbox_geom_src_crs], crop=True, all_touched=True,
            nodata=source_nodata_val if source_nodata_val is not None else 0 # Use 0 if no nodata, rio_mask requires a value
        )
    except ValueError as ve:
        if "Input shapes do not overlap raster." in str(ve):
            logging.warning(f"BBOX does not overlap raster {src_path}. Output will be minimal (1x1 pixel).")
            # Create a minimal 1x1 pixel image
            out_image = np.full((source_profile['count'], 1, 1),
                                 (source_nodata_val if source_nodata_val is not None else DEFAULT_NODATA_VALUE_FOR_OUTPUT_DTYPE),
                                 dtype=source_profile['dtype'])
            pixel_size_x = source_profile['transform'].a
            pixel_size_y = source_profile['transform'].e
            # Place the 1x1 pixel at the center of the (non-overlapping) BBOX centroid
            crop_center_x = bbox_geom_src_crs.centroid.x
            crop_center_y = bbox_geom_src_crs.centroid.y
            out_transform = rasterio.Affine(pixel_size_x, 0.0, crop_center_x - pixel_size_x / 2,
                                          0.0, pixel_size_y, crop_center_y - pixel_size_y / 2)
        else:
            raise ve

    # Handle cases where crop results in 0-dimension image (e.g. BBOX is a line on pixel boundary)
    if out_image.shape[1] == 0 or out_image.shape[2] == 0:
        h = 1 if out_image.shape[1] == 0 else out_image.shape[1]
        w = 1 if out_image.shape[2] == 0 else out_image.shape[2]
        logging.warning(f"Cropped image for {src_path} has near-zero width/height (shape: {out_image.shape}). Creating minimal {h}x{w} pixel output.")
        out_image = np.full((source_profile['count'], h, w),
                             (source_nodata_val if source_nodata_val is not None else DEFAULT_NODATA_VALUE_FOR_OUTPUT_DTYPE),
                             dtype=source_profile['dtype'])
        # If transform became invalid, try to center it if possible, or use a default
        if not out_transform or (h==1 and w==1):
            pixel_size_x = source_profile['transform'].a
            pixel_size_y = source_profile['transform'].e
            crop_center_x = bbox_geom_src_crs.centroid.x
            crop_center_y = bbox_geom_src_crs.centroid.y
            out_transform = rasterio.Affine(pixel_size_x, 0.0, crop_center_x - pixel_size_x / 2,
                                          0.0, pixel_size_y, crop_center_y - pixel_size_y / 2)


    out_meta = source_profile.copy()
    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "dtype": target_dtype,
        "nodata": target_nodata_val_for_output,
        "compress": "lzw" # Apply LZW compression
    })

    img_to_write = np.full(out_image.shape, target_nodata_val_for_output, dtype=target_dtype)
    if source_nodata_val is not None:
        # Create a mask of valid data based on source nodata
        # Ensure out_image is not empty and has the same number of bands as source_profile
        if out_image.ndim == 3 and out_image.shape[0] == source_profile['count']:
            valid_mask = np.ones(out_image.shape[1:], dtype=bool) # Start with all valid per pixel
            for i in range(out_image.shape[0]): # Check each band
                 # Handle potential float precision issues with nodata comparison if source_nodata_val is float
                if np.issubdtype(type(source_nodata_val), np.floating):
                    valid_mask &= ~np.isclose(out_image[i], source_nodata_val)
                else:
                    valid_mask &= (out_image[i] != source_nodata_val)
            
            # Apply mask across all bands
            for i in range(out_image.shape[0]):
                img_to_write[i][valid_mask] = out_image[i][valid_mask].astype(target_dtype)
        elif out_image.size > 0 : # Handle single band case if source_profile['count'] == 1
             if np.issubdtype(type(source_nodata_val), np.floating):
                valid_mask = ~np.isclose(out_image, source_nodata_val)
             else:
                valid_mask = (out_image != source_nodata_val)
             img_to_write[valid_mask] = out_image[valid_mask].astype(target_dtype)
        # else: out_image is empty or band mismatch, img_to_write remains all nodata
    else: # No source nodata, attempt to cast all
        img_to_write = out_image.astype(target_dtype)


    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(img_to_write)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logging.debug(f"Written {output_path}, size: {file_size_mb:.3f} MB")
    return file_size_mb


def determine_min_bbox_and_scale_for_size(
    src_path,
    initial_bbox_geom_src_crs, # Initial BBOX geometry already in source CRS
    source_profile,
    source_nodata_val,
    target_dtype,
    target_nodata_val_for_output,
    max_size_mb,
    temp_dir, # Temporary directory for intermediate files
    fallback_scale_factor=FALLBACK_SCALE_FACTOR,
    max_fallback_attempts=MAX_FALLBACK_ATTEMPTS,
    safety_margin=SAFETY_MARGIN_FACTOR
):
    """
    Determines the BBOX and side scaling factor needed for a single raster to meet size constraints.
    Uses a temporary file for size checking.
    Returns (final_bbox_geom_src_crs, effective_side_scale_factor, final_size_mb).
    effective_side_scale_factor is relative to the initial_bbox_geom_src_crs.
    """
    current_bbox_geom = initial_bbox_geom_src_crs
    effective_side_scale_factor = 1.0
    
    # Unique temp file name for this source file processing
    base_src_filename = os.path.splitext(os.path.basename(src_path))[0]
    temp_output_path = os.path.join(temp_dir, f"temp_{base_src_filename}.tif")

    logging.info(f"Processing {src_path} for size constraint with initial BBOX: {current_bbox_geom.bounds}")

    size_after_initial_crop_mb = _crop_and_write_raster(
        src_path, temp_output_path, current_bbox_geom,
        source_profile, source_nodata_val, target_dtype, target_nodata_val_for_output
    )
    logging.info(f"Size of {src_path} with initial BBOX crop: {size_after_initial_crop_mb:.2f} MB")

    if size_after_initial_crop_mb <= max_size_mb:
        logging.info(f"{src_path} with initial BBOX is within size limits.")
        os.remove(temp_output_path) # Clean up temp file
        return current_bbox_geom, 1.0, size_after_initial_crop_mb

    # Initial crop exceeds size limit, attempt direct scaling calculation
    logging.info(f"{src_path} initial crop exceeds limit. Attempting direct scaling.")
    
    # Calculate the side scale factor needed
    # Target slightly less than max_size_mb for safety
    size_ratio_for_area = (max_size_mb * safety_margin) / size_after_initial_crop_mb
    if size_ratio_for_area <= 0:
        logging.warning(f"Calculated size ratio for {src_path} is zero or negative. Will use a very small fallback scale.")
        # Use a drastic reduction factor, equivalent to multiple fallback steps
        side_scale_factor_direct = math.sqrt((fallback_scale_factor ** max_fallback_attempts)**2) # Effectively (fallback_scale_factor ** max_fallback_attempts)

    else:
        side_scale_factor_direct = math.sqrt(size_ratio_for_area)

    current_bbox_geom = _scale_bbox(initial_bbox_geom_src_crs, side_scale_factor_direct)
    effective_side_scale_factor = side_scale_factor_direct
    logging.info(f"Calculated scaled BBOX for {src_path}: {current_bbox_geom.bounds} with side_scale_factor {side_scale_factor_direct:.4f}")

    # Loop with fallback attempts, starting with the directly scaled BBOX
    for attempt in range(max_fallback_attempts + 1): # +1 for the initial scaled attempt
        if attempt > 0: # This is a fallback attempt
            logging.info(f"Fallback attempt {attempt}/{max_fallback_attempts} for {src_path}.")
            fb_side_scale_single_step = math.sqrt(fallback_scale_factor)
            current_bbox_geom = _scale_bbox(current_bbox_geom, fb_side_scale_single_step)
            effective_side_scale_factor *= fb_side_scale_single_step # Accumulate scaling
            logging.info(f"Fallback scaled BBOX for {src_path}: {current_bbox_geom.bounds}, new effective_side_scale_factor: {effective_side_scale_factor:.4f}")

        current_file_size_mb = _crop_and_write_raster(
            src_path, temp_output_path, current_bbox_geom,
            source_profile, source_nodata_val, target_dtype, target_nodata_val_for_output
        )
        logging.info(f"Size of {src_path} after scaling (attempt {attempt+1 if attempt >0 else 'direct'}): {current_file_size_mb:.2f} MB. BBOX ({source_profile['crs']}): {current_bbox_geom.bounds}")

        if current_file_size_mb <= max_size_mb:
            logging.info(f"Success for {src_path}: File size {current_file_size_mb:.2f}MB is within limit.")
            os.remove(temp_output_path) # Clean up temp file
            return current_bbox_geom, effective_side_scale_factor, current_file_size_mb
    
    # If loop finishes, all attempts failed to bring it under size
    logging.warning(f"Failed to reduce {src_path} below {max_size_mb}MB after all attempts. Smallest achieved: {current_file_size_mb:.2f}MB with effective_side_scale_factor: {effective_side_scale_factor:.4f}")
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)
    return current_bbox_geom, effective_side_scale_factor, current_file_size_mb


def create_forms_raster_excerpts_synced_bbox():
    """
    Creates clipped and compressed excerpts from FORMS height rasters.
    Pass 1: Determines a common BBOX scaling factor by checking all files.
    Pass 2: Applies this common BBOX (or original if no scaling needed) to all files.
    """
    logging.info("Starting FORMS raster excerpt creation with synchronized BBOX.")

    if not FORMS_RASTER_FILE_PATHS or not any(FORMS_RASTER_FILE_PATHS):
        logging.warning("No input raster files provided in FORMS_RASTER_FILE_PATHS. Exiting.")
        return

    # Create output directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_output_dir = os.path.join(script_dir, OUTPUT_RASTER_DIR_RELATIVE)
    os.makedirs(abs_output_dir, exist_ok=True)

    # Master BBOX in EPSG:2154
    master_initial_bbox_epsg2154 = box(*BBOX_COORDS_EPSG2154)
    master_bbox_gdf_epsg2154 = gpd.GeoDataFrame([{'geometry': master_initial_bbox_epsg2154}], crs=BBOX_CRS_EPSG2154)

    overall_min_effective_side_scale = 1.0
    temp_processing_dir = tempfile.mkdtemp(prefix="raster_excerpt_sizing_")
    logging.info(f"Created temporary directory for sizing: {temp_processing_dir}")

    # --- Pass 1: Determine the minimum common scaling factor ---
    logging.info("--- Pass 1: Determining minimum common BBOX scaling factor ---")
    processed_files_for_scaling = []

    for forms_raster_file_path in FORMS_RASTER_FILE_PATHS:
        if not os.path.exists(forms_raster_file_path):
            logging.error(f"ERROR: Input file not found: {forms_raster_file_path}. Skipping.")
            continue
        
        logging.info(f"Processing for BBOX scaling: {forms_raster_file_path}")
        try:
            source_profile, source_crs, _, source_nodata_val = _get_raster_info(forms_raster_file_path)
            
            # Transform master BBOX to this file's CRS
            if source_crs != rasterio.CRS.from_string(BBOX_CRS_EPSG2154):
                logging.debug(f"Reprojecting master BBOX from {BBOX_CRS_EPSG2154} to {source_crs} for {forms_raster_file_path}.")
                initial_cropping_gdf_src_crs = master_bbox_gdf_epsg2154.to_crs(source_crs)
            else:
                initial_cropping_gdf_src_crs = master_bbox_gdf_epsg2154.copy() # Use a copy
            initial_cropping_geom_src_crs = initial_cropping_gdf_src_crs.geometry.iloc[0]

            # Determine target_nodata_val for OUTPUT_DTYPE based on this specific file's nodata
            target_nodata_val = DEFAULT_NODATA_VALUE_FOR_OUTPUT_DTYPE
            if source_nodata_val is not None:
                try:
                    casted_source_nodata = np.array([source_nodata_val]).astype(OUTPUT_DTYPE)[0]
                    target_type_info = np.iinfo(OUTPUT_DTYPE) if np.issubdtype(np.dtype(OUTPUT_DTYPE), np.integer) else np.finfo(OUTPUT_DTYPE)
                    if float(casted_source_nodata) == float(source_nodata_val) and \
                       target_type_info.min <= casted_source_nodata <= target_type_info.max:
                        target_nodata_val = casted_source_nodata
                    # else: use default, warning will be logged by determine_min_bbox... if needed via _crop_and_write
                except (ValueError, OverflowError):
                    pass # Use default

            _, effective_side_scale, _ = determine_min_bbox_and_scale_for_size(
                src_path=forms_raster_file_path,
                initial_bbox_geom_src_crs=initial_cropping_geom_src_crs,
                source_profile=source_profile,
                source_nodata_val=source_nodata_val,
                target_dtype=OUTPUT_DTYPE,
                target_nodata_val_for_output=target_nodata_val,
                max_size_mb=HARD_MAX_SIZE_MB,
                temp_dir=temp_processing_dir
            )
            overall_min_effective_side_scale = min(overall_min_effective_side_scale, effective_side_scale)
            processed_files_for_scaling.append(forms_raster_file_path)
        except FileNotFoundError: # Should be caught by os.path.exists, but good practice
             logging.error(f"File not found during pass 1: {forms_raster_file_path}")
        except Exception as e:
            logging.error(f"Error during BBOX scaling determination for {forms_raster_file_path}: {e}", exc_info=True)
    
    if not processed_files_for_scaling:
        logging.error("No files were successfully processed for BBOX scaling. Cannot proceed.")
        shutil.rmtree(temp_processing_dir)
        logging.info(f"Removed temporary directory: {temp_processing_dir}")
        return

    logging.info(f"--- End Pass 1 --- Overall minimum effective side scale factor: {overall_min_effective_side_scale:.4f}")

    # Determine the final common BBOX in EPSG:2154
    if overall_min_effective_side_scale < 1.0:
        logging.info(f"Applying scaling factor {overall_min_effective_side_scale:.4f} to master BBOX {master_initial_bbox_epsg2154.bounds}")
        final_common_bbox_epsg2154 = _scale_bbox(master_initial_bbox_epsg2154, overall_min_effective_side_scale)
        logging.info(f"Final common scaled BBOX (EPSG:2154): {final_common_bbox_epsg2154.bounds}")
    else:
        final_common_bbox_epsg2154 = master_initial_bbox_epsg2154
        logging.info(f"No overall scaling needed. Using original master BBOX (EPSG:2154): {final_common_bbox_epsg2154.bounds}")
    
    final_common_bbox_gdf_epsg2154 = gpd.GeoDataFrame([{'geometry': final_common_bbox_epsg2154}], crs=BBOX_CRS_EPSG2154)

    # --- Pass 2: Create final excerpts using the common BBOX ---
    logging.info("--- Pass 2: Creating final excerpts with the common BBOX ---")
    for forms_raster_file_path in FORMS_RASTER_FILE_PATHS: # Iterate through original list to attempt all
        if not os.path.exists(forms_raster_file_path):
            logging.warning(f"Skipping {forms_raster_file_path} in Pass 2 as it was not found or skipped in Pass 1.")
            continue
        if forms_raster_file_path not in processed_files_for_scaling: # Skip if Pass 1 failed for this file
            logging.warning(f"Skipping {forms_raster_file_path} in Pass 2 as it was not successfully processed for scaling in Pass 1.")
            continue

        base_filename = os.path.splitext(os.path.basename(forms_raster_file_path))[0]
        # Dynamic output filename, e.g., excerpt_Height_mavg_2023.tif
        output_filename = f"excerpt_{base_filename}.tif"
        abs_output_path = os.path.join(abs_output_dir, output_filename)
        logging.info(f"Processing for final excerpt: {forms_raster_file_path} -> {abs_output_path}")

        try:
            source_profile, source_crs, _, source_nodata_val = _get_raster_info(forms_raster_file_path)

            # Transform final common BBOX to this file's CRS
            if source_crs != rasterio.CRS.from_string(BBOX_CRS_EPSG2154):
                logging.debug(f"Reprojecting final common BBOX from {BBOX_CRS_EPSG2154} to {source_crs} for {forms_raster_file_path}.")
                final_cropping_gdf_src_crs = final_common_bbox_gdf_epsg2154.to_crs(source_crs)
            else:
                final_cropping_gdf_src_crs = final_common_bbox_gdf_epsg2154.copy()
            final_cropping_geom_src_crs = final_cropping_gdf_src_crs.geometry.iloc[0]
            
            # Determine target_nodata_val again for this specific file
            target_nodata_val = DEFAULT_NODATA_VALUE_FOR_OUTPUT_DTYPE
            if source_nodata_val is not None:
                try:
                    casted_source_nodata = np.array([source_nodata_val]).astype(OUTPUT_DTYPE)[0]
                    target_type_info = np.iinfo(OUTPUT_DTYPE) if np.issubdtype(np.dtype(OUTPUT_DTYPE), np.integer) else np.finfo(OUTPUT_DTYPE)
                    if float(casted_source_nodata) == float(source_nodata_val) and \
                       target_type_info.min <= casted_source_nodata <= target_type_info.max:
                        target_nodata_val = casted_source_nodata
                except (ValueError, OverflowError):
                    pass # Use default

            final_size_mb = _crop_and_write_raster(
                src_path=forms_raster_file_path,
                output_path=abs_output_path,
                bbox_geom_src_crs=final_cropping_geom_src_crs,
                source_profile=source_profile,
                source_nodata_val=source_nodata_val,
                target_dtype=OUTPUT_DTYPE,
                target_nodata_val_for_output=target_nodata_val
            )

            if final_size_mb <= HARD_MAX_SIZE_MB:
                logging.info(f"SUCCESS: {output_filename} created. Size: {final_size_mb:.2f} MB. Used BBOX ({source_crs}): {final_cropping_geom_src_crs.bounds}")
            else:
                logging.critical(f"CRITICAL: {output_filename} size is {final_size_mb:.2f} MB, EXCEEDING {HARD_MAX_SIZE_MB} MB limit even with common BBOX. Used BBOX ({source_crs}): {final_cropping_geom_src_crs.bounds}")

        except Exception as e:
            logging.error(f"Error creating final excerpt for {forms_raster_file_path}: {e}", exc_info=True)

    # Clean up temporary directory
    try:
        shutil.rmtree(temp_processing_dir)
        logging.info(f"Successfully removed temporary directory: {temp_processing_dir}")
    except Exception as e:
        logging.error(f"Error removing temporary directory {temp_processing_dir}: {e}")

    logging.info("--- FORMS raster excerpt creation process finished. ---")


if __name__ == "__main__":
    # Ensure you have added the path to the 2022 file in FORMS_RASTER_FILE_PATHS
    # e.g. FORMS_RASTER_FILE_PATHS.append('/path/to/your/Height_mavg_2022.tif')
    if len(FORMS_RASTER_FILE_PATHS) < 2 : # Basic check
        logging.warning("FORMS_RASTER_FILE_PATHS contains fewer than 2 files. "
                        "Please add the path to the 2022 (and any other) FORMS files.")
        # Example of how to add another file if needed for testing:
        # if '/Users/arthurcalvi/Data/Disturbances_maps/FORMS/Height_mavg_2023.tif' in FORMS_RASTER_FILE_PATHS and \
        #    not any('2022' in f for f in FORMS_RASTER_FILE_PATHS):
        #     logging.info("Adding a placeholder for a 2022 file for demonstration if not present. Replace with actual path.")
        #     # FORMS_RASTER_FILE_PATHS.append('/Users/arthurcalvi/Data/Disturbances_maps/FORMS/Height_mavg_2022.tif') # Replace this
    
    create_forms_raster_excerpts_synced_bbox() 