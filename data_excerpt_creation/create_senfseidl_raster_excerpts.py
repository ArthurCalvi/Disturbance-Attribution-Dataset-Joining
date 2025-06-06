import os
import logging
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.crs import CRS as RasterioCRS # Renamed to avoid conflict
from shapely.geometry import box
import geopandas as gpd
import numpy as np
import argparse
import sys

# Add src to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(script_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from config.constants import EXCERPT_BOUNDING_BOXES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
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
OUTPUT_DIR_RASTERS = "../excerpts/raw/senfseidl"

TIFF_COMPRESSION = "LZW" # Added compression

# --- Helper Functions ---

def crop_and_save_raster(raster_path_info, output_dir_relative, bbox_coords, bbox_crs_str, bbox_name):
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

    output_filename = f"excerpt_{bbox_name}_{os.path.basename(input_raster_path_relative)}"
    abs_output_raster_path = os.path.abspath(os.path.join(script_dir, output_dir_relative, output_filename))
    os.makedirs(os.path.dirname(abs_output_raster_path), exist_ok=True)

    logging.debug(f"Processing {dataset_name}: {abs_input_raster_path}")

    try:
        with rasterio.open(abs_input_raster_path) as src:
            raster_crs = src.crs
            logging.debug(f"Source raster CRS for {dataset_name}: {raster_crs}, Data type: {src.dtypes[0]}")

            if not raster_crs:
                if expected_crs_if_missing_str:
                    logging.warning(f"CRS for {dataset_name} is not set in the file. Assuming {expected_crs_if_missing_str} as per configuration.")
                    raster_crs = RasterioCRS.from_string(expected_crs_if_missing_str)
                else:
                    logging.error(f"CRS for {dataset_name} is not set in the file and no fallback is configured. Skipping.")
                    return False
            
            shapely_bbox = box(*bbox_coords)
            bbox_gdf = gpd.GeoDataFrame([{'id': 1, 'geometry': shapely_bbox}], crs=bbox_crs_str)

            logging.debug(f"Reprojecting BBOX from {bbox_crs_str} to {raster_crs} for {dataset_name}.")
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
                logging.debug(f"Converting data type from {src.dtypes[0]} to {output_dtype_str} for {dataset_name}.")
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

            logging.debug(f"Saving cropped raster for {dataset_name} to {abs_output_raster_path} with {TIFF_COMPRESSION} compression and dtype {output_dtype_str}.")
            with rasterio.open(abs_output_raster_path, "w", **out_meta) as dest:
                dest.write(out_image)
            
            file_size_mb = os.path.getsize(abs_output_raster_path) / (1024 * 1024)
            logging.info(f"Successfully created excerpt: {os.path.basename(abs_output_raster_path)}, size {file_size_mb:.2f} MB")

            return True

    except Exception as e:
        logging.error(f"Error processing raster {dataset_name} ({abs_input_raster_path}): {e}", exc_info=True)
        return False

def main(bbox_name):
    """
    Main function to create Senf & Seidl raster excerpts.
    """
    logging.debug(f"Starting SenfSeidl raw raster excerpt creation using bbox: '{bbox_name}'...")
    
    selected_bbox = EXCERPT_BOUNDING_BOXES[bbox_name]
    bbox_coords = selected_bbox['coords']
    bbox_crs = selected_bbox['crs']
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_output_dir = os.path.abspath(os.path.join(script_dir, OUTPUT_DIR_RASTERS))
    os.makedirs(abs_output_dir, exist_ok=True)
    logging.debug(f"Output directory for raster excerpts: {abs_output_dir}")

    # Sanitize bbox_name for the filename
    bbox_name_fs = bbox_name.replace(' ', '_')

    processed_files = 0
    failed_files = 0
    for raster_info in RAW_RASTER_FILES:
        if crop_and_save_raster(raster_info, OUTPUT_DIR_RASTERS, bbox_coords, bbox_crs, bbox_name_fs):
            processed_files += 1
        else:
            failed_files += 1
    
    logging.info("--- Summary ---")
    logging.info(f"Total raw raster files processed: {len(RAW_RASTER_FILES)}")
    logging.info(f"Successfully created excerpts: {processed_files}")
    logging.info(f"Failed/skipped excerpts: {failed_files}")
    logging.debug("Raw raster excerpt creation process finished.")

# --- Main Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create raster excerpts for Senf & Seidl data.")
    parser.add_argument(
        '--bbox-name',
        type=str,
        default='les landes',
        choices=EXCERPT_BOUNDING_BOXES.keys(),
        help=f"The name of the bounding box to use. Defined in src/config/constants.py. "
             f"Choices: {list(EXCERPT_BOUNDING_BOXES.keys())}"
    )
    args = parser.parse_args()
    main(args.bbox_name) 