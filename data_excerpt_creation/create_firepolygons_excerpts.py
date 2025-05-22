import os
import logging
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import shutil # For copying the CSV

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Path to the external CSV file
FIRE_CSV_FILE_PATH = '/Users/arthurcalvi/Data/Disturbances_maps/FirePolygons/FFUD_Arthur/FFUD_Inventory_Arthur.csv'

# Path to the directory containing individual fire polygon GPKG files
FIRE_GPKG_DIR_PATH = '/Users/arthurcalvi/Data/Disturbances_maps/FirePolygons/FFUD_Arthur/Fire_folder/'

# BBOX coordinates (minx, miny, maxx, maxy) in EPSG:2154
BBOX_COORDS_EPSG2154 = (307783.0822, 6340505.4366, 469246.8845, 6419190.9011)
TARGET_CRS_EPSG2154 = "EPSG:2154"

# Output directories relative to this script's location
OUTPUT_GPKG_DIR_RELATIVE = "../excerpts/raw/firepolygons_gpkg/"
OUTPUT_CSV_DIR_RELATIVE = "../excerpts/raw/"
OUTPUT_CSV_FILENAME = "FFUD_Inventory_Arthur_excerpt.csv" # Copied CSV

# File size limits in MB
MAX_SIZE_MB_WARN = 5.0
MAX_SIZE_MB_CRITICAL = 10.0

def create_gpkg_excerpts():
    logging.info("--- Starting Fire Polygon GPKG Excerpt Creation ---")
    abs_output_gpkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), OUTPUT_GPKG_DIR_RELATIVE))
    os.makedirs(abs_output_gpkg_dir, exist_ok=True)
    logging.info(f"Output directory for GPKG excerpts: {abs_output_gpkg_dir}")

    bbox_geom_epsg2154 = box(*BBOX_COORDS_EPSG2154)
    # Create a GeoDataFrame for the bbox to use in geopandas.clip
    bbox_gdf_epsg2154 = gpd.GeoDataFrame([{'geometry': bbox_geom_epsg2154}], crs=TARGET_CRS_EPSG2154)

    if not os.path.isdir(FIRE_GPKG_DIR_PATH):
        logging.error(f"Fire GPKG directory not found: {FIRE_GPKG_DIR_PATH}")
        return

    gpkg_files_processed = 0
    gpkg_files_skipped_error = 0
    gpkg_files_empty_clip = 0

    for filename in os.listdir(FIRE_GPKG_DIR_PATH):
        if filename.lower().endswith(".gpkg"):
            file_path = os.path.join(FIRE_GPKG_DIR_PATH, filename)
            output_path = os.path.join(abs_output_gpkg_dir, filename)
            logging.info(f"Processing GPKG: {file_path}")

            try:
                gdf = gpd.read_file(file_path)
                
                # Ensure correct CRS before clipping
                if gdf.crs is None:
                    logging.warning(f"CRS for {filename} is missing. Assuming EPSG:2154 based on notebook context.")
                    gdf.crs = TARGET_CRS_EPSG2154 # Common case in notebook
                elif gdf.crs.to_string() != TARGET_CRS_EPSG2154:
                    logging.info(f"Reprojecting {filename} from {gdf.crs.to_string()} to {TARGET_CRS_EPSG2154}")
                    gdf = gdf.to_crs(TARGET_CRS_EPSG2154)

                clipped_gdf = gpd.clip(gdf, bbox_gdf_epsg2154, keep_geom_type=True)

                if clipped_gdf.empty:
                    logging.info(f"No features in {filename} intersected the BBOX. Skipping save.")
                    gpkg_files_empty_clip += 1
                    continue

                clipped_gdf.to_file(output_path, driver="GPKG")
                gpkg_files_processed += 1
                logging.info(f"Saved clipped GPKG to {output_path}")

                # File size check
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                if file_size_mb > MAX_SIZE_MB_CRITICAL:
                    logging.critical(f"CRITICAL: Clipped file {output_path} is {file_size_mb:.2f}MB ( > {MAX_SIZE_MB_CRITICAL}MB).")
                elif file_size_mb > MAX_SIZE_MB_WARN:
                    logging.warning(f"WARNING: Clipped file {output_path} is {file_size_mb:.2f}MB ( > {MAX_SIZE_MB_WARN}MB).")

            except Exception as e:
                logging.error(f"Error processing {filename}: {e}", exc_info=True)
                gpkg_files_skipped_error +=1

    logging.info("--- Fire Polygon GPKG Excerpt Creation Summary ---")
    logging.info(f"Successfully processed and saved: {gpkg_files_processed} GPKG files.")
    logging.info(f"Skipped due to errors: {gpkg_files_skipped_error} GPKG files.")
    logging.info(f"Skipped due to no features in BBOX: {gpkg_files_empty_clip} GPKG files.")
    logging.info("-------------------------------------------------")


def copy_csv_metadata():
    logging.info("--- Starting Fire Polygon CSV Metadata Copy ---")
    abs_output_csv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), OUTPUT_CSV_DIR_RELATIVE))
    os.makedirs(abs_output_csv_dir, exist_ok=True)
    
    abs_output_csv_path = os.path.join(abs_output_csv_dir, OUTPUT_CSV_FILENAME)

    if not os.path.exists(FIRE_CSV_FILE_PATH):
        logging.error(f"Source CSV file not found: {FIRE_CSV_FILE_PATH}")
        return

    try:
        shutil.copy(FIRE_CSV_FILE_PATH, abs_output_csv_path)
        logging.info(f"Successfully copied CSV metadata to: {abs_output_csv_path}")
        
        # File size check for the copied CSV
        file_size_mb = os.path.getsize(abs_output_csv_path) / (1024 * 1024)
        if file_size_mb > MAX_SIZE_MB_CRITICAL: # Using same limits, though less likely for a metadata CSV
            logging.critical(f"CRITICAL: Copied CSV file {abs_output_csv_path} is {file_size_mb:.2f}MB ( > {MAX_SIZE_MB_CRITICAL}MB).")
        elif file_size_mb > MAX_SIZE_MB_WARN:
            logging.warning(f"WARNING: Copied CSV file {abs_output_csv_path} is {file_size_mb:.2f}MB ( > {MAX_SIZE_MB_WARN}MB).")

    except Exception as e:
        logging.error(f"Error copying CSV file: {e}", exc_info=True)
    
    logging.info("--- Fire Polygon CSV Metadata Copy Finished ---")


if __name__ == "__main__":
    logging.info("Starting fire polygon excerpt creation process...")
    create_gpkg_excerpts()
    copy_csv_metadata()
    logging.info("Fire polygon excerpt creation process finished.") 