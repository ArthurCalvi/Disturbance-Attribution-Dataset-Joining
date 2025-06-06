import os
import logging
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom
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
# Path to the directory containing consolidated CDI GeoTIFF files (e.g., cdi_yyyyMMdd.tif)
CDI_RASTER_DIR_PATH = '/Users/arthurcalvi/Data/Disturbances_maps/Copernicus_CDI/CDI_2012_2023/'

EXPECTED_RASTER_CRS_EPSG3035 = "EPSG:3035"

# Output directory relative to this script's location
OUTPUT_DIR_RELATIVE = "../excerpts/raw/cdi/"
OUTPUT_DTYPE = "uint8"  # CDI values are typically small integers (0-4)
NODATA_VAL = 255         # Using a common nodata value for uint8 if original is different or not set

def create_cdi_raster_excerpt(
    input_raster_path,
    output_raster_path,
    bbox_geom,
    bbox_crs
):
    """Crops a single CDI raster file to a predefined BBOX and saves it as an excerpt."""
    logging.info(f"Processing: {input_raster_path}")
    try:
        with rasterio.open(input_raster_path) as src:
            raster_crs = src.crs
            if not raster_crs:
                logging.warning(f"Raster {os.path.basename(input_raster_path)} has no CRS. Assuming {EXPECTED_RASTER_CRS_EPSG3035}.")
                raster_crs = rasterio.CRS.from_string(EXPECTED_RASTER_CRS_EPSG3035)
            
            if str(raster_crs).upper() != EXPECTED_RASTER_CRS_EPSG3035.upper():
                logging.warning(f"Raster {os.path.basename(input_raster_path)} CRS is {raster_crs}, expected {EXPECTED_RASTER_CRS_EPSG3035}. Proceeding with transformation.")

            # Transform BBOX from its CRS to the raster's CRS
            bbox_gdf = gpd.GeoDataFrame([{'geometry': bbox_geom}], crs=bbox_crs)
            bbox_geom_in_src_crs = bbox_gdf.to_crs(raster_crs).iloc[0].geometry
            
            # Crop the raster
            out_image, out_transform = rio_mask(src, [bbox_geom_in_src_crs], crop=True, nodata=src.nodata if src.nodata is not None else NODATA_VAL)
            out_meta = src.meta.copy()

            # Update metadata for the cropped raster
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "crs": raster_crs,
                "dtype": OUTPUT_DTYPE,
                "nodata": NODATA_VAL,
                "compress": "lzw"
            })
            
            if src.nodata is not None and src.nodata != NODATA_VAL:
                out_image[out_image == src.nodata] = NODATA_VAL
            
            out_image_typed = out_image.astype(OUTPUT_DTYPE)

            with rasterio.open(output_raster_path, "w", **out_meta) as dest:
                dest.write(out_image_typed)
            
            file_size_mb = os.path.getsize(output_raster_path) / (1024 * 1024)
            logging.debug(f"Saved excerpt: {output_raster_path}, size: {file_size_mb:.2f} MB")
            return True

    except Exception as e:
        logging.error(f"Failed to process {input_raster_path}: {e}", exc_info=True)
        return False

def main(bbox_name):
    """
    Main function to create CDI raster excerpts.
    """
    logging.debug(f"Starting CDI raster excerpt creation using bbox: '{bbox_name}'.")
    
    selected_bbox = EXCERPT_BOUNDING_BOXES[bbox_name]
    bbox_geom = box(*selected_bbox['coords'])
    bbox_crs = selected_bbox['crs']

    # Create the absolute path for the output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_output_dir = os.path.abspath(os.path.join(script_dir, OUTPUT_DIR_RELATIVE))
    os.makedirs(abs_output_dir, exist_ok=True)
    logging.debug(f"Output directory: {abs_output_dir}")

    if not os.path.isdir(CDI_RASTER_DIR_PATH):
        logging.error(f"Source directory not found: {CDI_RASTER_DIR_PATH}")
        sys.exit(1)

    processed_files = 0
    failed_files = 0
    for filename in os.listdir(CDI_RASTER_DIR_PATH):
        if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
            input_raster_path = os.path.join(CDI_RASTER_DIR_PATH, filename)
            
            # Sanitize bbox_name for the filename
            bbox_name_fs = bbox_name.replace(' ', '_')
            output_raster_name = f"excerpt_{bbox_name_fs}_{filename}"
            output_raster_path = os.path.join(abs_output_dir, output_raster_name)

            if create_cdi_raster_excerpt(input_raster_path, output_raster_path, bbox_geom, bbox_crs):
                processed_files += 1
            else:
                failed_files += 1
        else:
            logging.debug(f"Skipping non-TIFF file: {filename}")

    logging.info("--- Summary ---")
    if processed_files == 0 and failed_files == 0:
        logging.warning(f"No TIFF files found in {CDI_RASTER_DIR_PATH}.")
    else:
        logging.info(f"Successfully created {processed_files} excerpts.")
        logging.info(f"Failed to create {failed_files} excerpts.")
    logging.debug("CDI raster excerpt creation finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create raster excerpts for CDI data based on a named bounding box.")
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