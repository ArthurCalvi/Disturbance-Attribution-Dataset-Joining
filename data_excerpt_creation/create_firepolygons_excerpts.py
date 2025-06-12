import os
import logging
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import shutil # For copying the CSV
import argparse
import sys
from tqdm import tqdm

# Add src to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(script_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from config.constants import EXCERPT_BOUNDING_BOXES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Suppress noisy INFO logs from fiona, used by geopandas
logging.getLogger('fiona').setLevel(logging.WARNING)

# --- Constants ---
# Path to the external CSV file
FIRE_CSV_FILE_PATH = '/Users/arthurcalvi/Data/Disturbances_maps/FirePolygons/FFUD_Arthur/FFUD_Inventory_Arthur.csv'

# Path to the directory containing individual fire polygon GPKG files
FIRE_GPKG_DIR_PATH = '/Users/arthurcalvi/Data/Disturbances_maps/FirePolygons/FFUD_Arthur/Fire_folder/'

TARGET_CRS_EPSG2154 = "EPSG:2154"

# Output directories relative to this script's location
OUTPUT_GPKG_DIR_RELATIVE = "../excerpts/raw/firepolygons_gpkg/"
OUTPUT_CSV_DIR_RELATIVE = "../excerpts/raw/"
OUTPUT_CSV_FILENAME = "FFUD_Inventory_Arthur_excerpt.csv" # Copied CSV

def create_gpkg_excerpts(bbox_name):
    logging.debug("--- Starting Fire Polygon GPKG Excerpt Creation ---")
    abs_output_gpkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), OUTPUT_GPKG_DIR_RELATIVE))
    os.makedirs(abs_output_gpkg_dir, exist_ok=True)
    logging.debug(f"Output directory for GPKG excerpts: {abs_output_gpkg_dir}")

    selected_bbox = EXCERPT_BOUNDING_BOXES[bbox_name]
    bbox_coords = selected_bbox['coords']
    bbox_crs = selected_bbox['crs']
    
    bbox_geom = box(*bbox_coords)
    # Create a GeoDataFrame for the bbox to use in geopandas.clip
    bbox_gdf = gpd.GeoDataFrame([{'geometry': bbox_geom}], crs=bbox_crs)

    if not os.path.isdir(FIRE_GPKG_DIR_PATH):
        logging.error(f"Fire GPKG directory not found: {FIRE_GPKG_DIR_PATH}")
        return

    gpkg_files_processed = 0
    gpkg_files_skipped_error = 0
    gpkg_files_empty_clip = 0

    gpkg_files = [f for f in os.listdir(FIRE_GPKG_DIR_PATH) if f.lower().endswith(".gpkg")]

    for filename in tqdm(gpkg_files, desc="Clipping fire polygons", unit="file"):
        file_path = os.path.join(FIRE_GPKG_DIR_PATH, filename)

        # Sanitize bbox_name for the filename
        bbox_name_fs = bbox_name.replace(' ', '_')
        output_path = os.path.join(abs_output_gpkg_dir, f"excerpt_{bbox_name_fs}_{filename}")
        logging.debug(f"Processing GPKG: {file_path}")

        try:
            gdf = gpd.read_file(file_path)
            
            # Ensure correct CRS before clipping
            if gdf.crs is None:
                logging.warning(f"CRS for {filename} is missing. Assuming EPSG:2154 based on notebook context.")
                gdf.crs = TARGET_CRS_EPSG2154 # Common case in notebook
            elif gdf.crs.to_string() != bbox_crs:
                logging.debug(f"Reprojecting {filename} from {gdf.crs.to_string()} to {bbox_crs}")
                gdf = gdf.to_crs(bbox_crs)

            clipped_gdf = gpd.clip(gdf, bbox_gdf, keep_geom_type=True)

            if clipped_gdf.empty:
                logging.debug(f"No features in {filename} intersected the BBOX. Skipping save.")
                gpkg_files_empty_clip += 1
                continue

            clipped_gdf.to_file(output_path, driver="GPKG")
            gpkg_files_processed += 1
            logging.debug(f"Saved clipped GPKG to {output_path}")

        except Exception as e:
            logging.error(f"Error processing {filename}: {e}", exc_info=True)
            gpkg_files_skipped_error +=1

    logging.info("--- Fire Polygon GPKG Excerpt Creation Summary ---")
    logging.info(f"Successfully processed and saved: {gpkg_files_processed} GPKG files.")
    logging.info(f"Skipped due to errors: {gpkg_files_skipped_error} GPKG files.")
    logging.info(f"Skipped due to no features in BBOX: {gpkg_files_empty_clip} GPKG files.")
    logging.info("-------------------------------------------------")


def copy_csv_metadata():
    logging.debug("--- Starting Fire Polygon CSV Metadata Copy ---")
    abs_output_csv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), OUTPUT_CSV_DIR_RELATIVE))
    os.makedirs(abs_output_csv_dir, exist_ok=True)
    
    abs_output_csv_path = os.path.join(abs_output_csv_dir, OUTPUT_CSV_FILENAME)

    if not os.path.exists(FIRE_CSV_FILE_PATH):
        logging.error(f"Source CSV file not found: {FIRE_CSV_FILE_PATH}")
        return

    try:
        shutil.copy(FIRE_CSV_FILE_PATH, abs_output_csv_path)
        logging.debug(f"Successfully copied CSV metadata to: {abs_output_csv_path}")
        
    except Exception as e:
        logging.error(f"Error copying CSV file: {e}", exc_info=True)
    
    logging.debug("--- Fire Polygon CSV Metadata Copy Finished ---")


def main(bbox_name):
    """
    Main function to run the fire polygon excerpt creation process.
    """
    logging.debug(f"Starting fire polygon excerpt creation process using bbox: '{bbox_name}'...")
    create_gpkg_excerpts(bbox_name)
    copy_csv_metadata()
    logging.debug("Fire polygon excerpt creation process finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create excerpts for FirePolygon data (GPKG files and metadata CSV).")
    parser.add_argument(
        '--bbox-name',
        type=str,
        default='les landes',
        choices=EXCERPT_BOUNDING_BOXES.keys(),
        help=f"The name of the bounding box to use for clipping GPKGs. "
             f"Defined in src/config/constants.py. Choices: {list(EXCERPT_BOUNDING_BOXES.keys())}"
    )
    args = parser.parse_args()
    main(args.bbox_name) 