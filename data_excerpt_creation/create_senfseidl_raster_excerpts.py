import os
import logging
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.crs import CRS as RasterioCRS # Renamed to avoid conflict
from shapely.geometry import box
import geopandas as gpd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# BBOX coordinates (minx, miny, maxx, maxy) from create_data_excerpts.py
BBOX_COORDS_EPSG2154 = (307783.0822, 6340505.4366, 469246.8845, 6419190.9011)
BBOX_CRS_EPSG2154 = "EPSG:2154"

# Relative paths from the script's location (data_excerpt_creation/)
# to the raw raster files.
RAW_RASTER_FILES = [
    {
        "name": "SenfSeidl_Cause",
        "path": "../data/SenfSeidl_maps/fire_wind_barkbeetle_france.tif",
        "expected_crs_if_missing": "EPSG:3035",
        "output_dtype": "uint8",
        "nodata_val": 0 # Nodata for cause map if converted to uint8, assuming 0 is not a valid cause
    },
    {
        "name": "SenfSeidl_Year",
        "path": "../data/SenfSeidl_maps/france/disturbance_year_1986-2020_france.tif",
        "expected_crs_if_missing": None,
        "output_dtype": "uint16",
        "nodata_val": 65535 # Nodata for year map is 65535
    }
]

# Output directory for the raster excerpts, relative to this script's location.
OUTPUT_DIR_RASTERS = "../excerpts/raw/"

# Maximum file size in MB before issuing warnings
MAX_SIZE_MB_WARN = 5.0
MAX_SIZE_MB_CRITICAL = 10.0
TIFF_COMPRESSION = "LZW" # Added compression

# --- Helper Functions ---

def crop_and_save_raster(raster_path_info, output_dir_relative, bbox_coords, bbox_crs_str):
    """
    Crops a raster to the given bounding box and saves it.
    The bounding box is reprojected to the raster's native CRS before cropping.
    """
    input_raster_path_relative = raster_path_info["path"]
    dataset_name = raster_path_info["name"]
    expected_crs_if_missing_str = raster_path_info["expected_crs_if_missing"]
    output_dtype_str = raster_path_info["output_dtype"]
    nodata_val = raster_path_info["nodata_val"]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_input_raster_path = os.path.abspath(os.path.join(script_dir, input_raster_path_relative))

    if not os.path.exists(abs_input_raster_path):
        logging.error(f"Input raster file not found: {abs_input_raster_path}. Skipping {dataset_name}.")
        return False

    output_filename = f"excerpt_{os.path.basename(input_raster_path_relative)}"
    abs_output_raster_path = os.path.abspath(os.path.join(script_dir, output_dir_relative, output_filename))
    os.makedirs(os.path.dirname(abs_output_raster_path), exist_ok=True)

    logging.info(f"Processing {dataset_name}: {abs_input_raster_path}")

    try:
        with rasterio.open(abs_input_raster_path) as src:
            raster_crs = src.crs
            logging.info(f"Source raster CRS for {dataset_name}: {raster_crs}, Data type: {src.dtypes[0]}")

            if not raster_crs:
                if expected_crs_if_missing_str:
                    logging.warning(f"CRS for {dataset_name} is not set in the file. Assuming {expected_crs_if_missing_str} as per configuration.")
                    raster_crs = RasterioCRS.from_string(expected_crs_if_missing_str)
                else:
                    logging.error(f"CRS for {dataset_name} is not set in the file and no fallback is configured. Skipping.")
                    return False
            
            shapely_bbox = box(*bbox_coords)
            bbox_gdf = gpd.GeoDataFrame([{'id': 1, 'geometry': shapely_bbox}], crs=bbox_crs_str)

            logging.info(f"Reprojecting BBOX from {bbox_crs_str} to {raster_crs} for {dataset_name}.")
            try:
                bbox_gdf_reprojected = bbox_gdf.to_crs(raster_crs)
            except Exception as e:
                logging.error(f"Failed to reproject BBOX for {dataset_name} to CRS {raster_crs}. Error: {e}")
                return False
            
            cropping_geometry = [bbox_gdf_reprojected.geometry.iloc[0]]
            
            # Determine nodata value for masking. Use source nodata if available and not changing type, else use defined nodata_val
            # If converting type, the original nodata might not be valid for the new type or concept (e.g. float nodata for int type)
            mask_nodata = src.nodata if src.nodata is not None and src.dtypes[0] == output_dtype_str else nodata_val
            # If output dtype is float, ensure nodata_val is also float if it's an integer for consistency
            if np.issubdtype(np.dtype(output_dtype_str), np.floating) and np.issubdtype(type(nodata_val), np.integer):
                nodata_val_for_meta = float(nodata_val)
            else:
                nodata_val_for_meta = nodata_val

            out_image, out_transform = rio_mask(src, cropping_geometry, crop=True, nodata=mask_nodata, filled=True)
            
            # Change data type if needed
            if src.dtypes[0] != output_dtype_str:
                logging.info(f"Converting data type from {src.dtypes[0]} to {output_dtype_str} for {dataset_name}.")
                # Handle potential NaN conversion if original is float and target is int
                if np.issubdtype(src.dtypes[0], np.floating) and np.issubdtype(np.dtype(output_dtype_str), np.integer):
                    # Replace NaNs with the integer nodata value before casting
                    out_image = np.nan_to_num(out_image, nan=nodata_val_for_meta)
                out_image = out_image.astype(output_dtype_str)

            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff", # Explicitly set to GTiff for compression options
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "crs": raster_crs,
                "dtype": output_dtype_str, # Update dtype in metadata
                "nodata": nodata_val_for_meta, # Set nodata value for the output type
                "compress": TIFF_COMPRESSION # Added compression
            })
            out_meta['count'] = out_image.shape[0]

            logging.info(f"Saving cropped raster for {dataset_name} to {abs_output_raster_path} with {TIFF_COMPRESSION} compression and dtype {output_dtype_str}.")
            with rasterio.open(abs_output_raster_path, "w", **out_meta) as dest:
                dest.write(out_image)
            
            logging.info(f"Successfully created excerpt: {abs_output_raster_path}")

            file_size_bytes = os.path.getsize(abs_output_raster_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            logging.info(f"Excerpt size for {dataset_name}: {file_size_mb:.2f} MB")

            if file_size_mb > MAX_SIZE_MB_CRITICAL:
                logging.critical(
                    f"Excerpt {abs_output_raster_path} ({file_size_mb:.2f} MB) is larger than {MAX_SIZE_MB_CRITICAL}MB. "
                    "Consider different compression or resolution reduction if still too large."
                )
            elif file_size_mb > MAX_SIZE_MB_WARN:
                logging.warning(
                    f"Excerpt {abs_output_raster_path} ({file_size_mb:.2f} MB) is larger than {MAX_SIZE_MB_WARN}MB. "
                )
            return True

    except Exception as e:
        logging.error(f"Error processing raster {dataset_name} ({abs_input_raster_path}): {e}", exc_info=True)
        return False

# --- Main Script ---
if __name__ == "__main__":
    logging.info("Starting raw raster excerpt creation process with compression and dtype optimization...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_output_dir = os.path.abspath(os.path.join(script_dir, OUTPUT_DIR_RASTERS))
    os.makedirs(abs_output_dir, exist_ok=True)
    logging.info(f"Output directory for raster excerpts: {abs_output_dir}")

    success_count = 0
    fail_count = 0

    for raster_info in RAW_RASTER_FILES:
        if crop_and_save_raster(raster_info, OUTPUT_DIR_RASTERS, BBOX_COORDS_EPSG2154, BBOX_CRS_EPSG2154):
            success_count += 1
        else:
            fail_count += 1
    
    logging.info("--- Summary ---")
    logging.info(f"Total raw raster files processed: {len(RAW_RASTER_FILES)}")
    logging.info(f"Successfully created excerpts: {success_count}")
    logging.info(f"Failed/skipped excerpts: {fail_count}")
    logging.info("Raw raster excerpt creation process finished.") 