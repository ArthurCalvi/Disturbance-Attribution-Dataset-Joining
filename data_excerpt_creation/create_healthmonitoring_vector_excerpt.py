import os
import logging
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import numpy as np
import pyarrow # Added for specific error catching
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
# Path to the external Excel file
EXCEL_FILE_PATH = '/Users/arthurcalvi/Data/Disturbances_maps/Thierry Belouard & DSF/Veille_sanitaire/veille sanitaire DSF 2007_2023.xlsx'
EXCEL_SHEET_NAME = 'signalement0'
EXCEL_HEADER_ROW = 2 # Header is on the 3rd row, so index 2
LAT_COL = 'Latitude'
LON_COL = 'Longitude'
PROBLEM_COLUMN_HANDLING = { # Columns that need specific dtype handling
    "Essence dominante": str,
    "Essence dominante (code)": str, # Proactively handle similar columns if they exist
    "Code agent pathogène": str,
    "Code agent pathogene": str, # handle potential variations
    "Essence concernée": str, # Added to handle mixed types for Parquet export
    "Essence regroupée (ess. concernée)": str, # Added to handle mixed types for Parquet export
    # Add other problematic columns here with their desired type if encountered
}

INITIAL_DATA_CRS_EPSG4326 = "EPSG:4326"

# Output directory and filename, relative to this script's location
OUTPUT_DIR_RELATIVE = "../excerpts/raw/hm/"

def main(bbox_name):
    """
    Reads the Health Monitoring Excel file, creates a GeoDataFrame,
    clips it to the BBOX, handles problematic data types, and saves it as Parquet.
    """
    logging.debug(f"Starting Health Monitoring data excerpt creation process for BBOX: '{bbox_name}'...")
    
    # Sanitize bbox_name for the filename
    bbox_name_fs = bbox_name.replace(' ', '_')
    output_filename = f"hm_excerpt_{bbox_name_fs}.gpkg"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_output_dir = os.path.abspath(os.path.join(script_dir, OUTPUT_DIR_RELATIVE))
    os.makedirs(abs_output_dir, exist_ok=True)
    abs_output_path = os.path.join(abs_output_dir, output_filename)

    if os.path.exists(abs_output_path):
        logging.info(f"Output file already exists, skipping: {abs_output_path}")
        return True # Consider it a success if the file is already there

    # Get BBOX details
    selected_bbox = EXCERPT_BOUNDING_BOXES.get(bbox_name)
    if selected_bbox is None:
        logging.error(f"BBOX '{bbox_name}' not found in EXCERPT_BOUNDING_BOXES. Aborting.")
        return False
    bbox_coords = selected_bbox['coords']
    target_crs = selected_bbox['crs']

    if not os.path.exists(EXCEL_FILE_PATH):
        logging.error(f"Source Excel file not found: {EXCEL_FILE_PATH}. Aborting.")
        return False

    try:
        logging.debug(f"Attempting to load Excel file: {EXCEL_FILE_PATH}, sheet: {EXCEL_SHEET_NAME}, header row: {EXCEL_HEADER_ROW}")
        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=EXCEL_SHEET_NAME, header=EXCEL_HEADER_ROW)
        logging.debug(f"Successfully loaded Excel data. {len(df)} rows found.")

        # Data Cleaning & Preparation
        df.dropna(subset=[LON_COL, LAT_COL], inplace=True)
        logging.debug(f"{len(df)} rows remaining after dropping NA in lat/lon.")
        
        # Initial explicit conversion for known problematic columns
        for col, dtype in PROBLEM_COLUMN_HANDLING.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                    logging.debug(f"Successfully converted column '{col}' to {dtype}.")
                except Exception as e:
                    logging.warning(f"Could not convert column '{col}' to {dtype}: {e}. Skipping conversion for this column.")
            else:
                logging.warning(f"Problematic column '{col}' configured for handling not found in DataFrame.")


        geometry = [Point(xy) for xy in zip(df[LON_COL], df[LAT_COL])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=INITIAL_DATA_CRS_EPSG4326)
        logging.debug(f"Successfully converted to GeoDataFrame with initial CRS: {INITIAL_DATA_CRS_EPSG4326}")

        logging.debug(f"Reprojecting GeoDataFrame from {INITIAL_DATA_CRS_EPSG4326} to {target_crs}.")
        gdf = gdf.to_crs(target_crs)

        logging.debug(f"Clipping GeoDataFrame to BBOX in {target_crs}.")
        bbox_polygon = box(*bbox_coords)
        clipped_gdf = gdf[gdf.geometry.intersects(bbox_polygon)].copy()
        
        if clipped_gdf.empty:
            logging.warning("No features found within the BBOX. Output file will not be created.")
            return False
        
        logging.info(f"{len(clipped_gdf)} features found within BBOX.")

        # --- Robust Parquet Saving --- 
        parquet_saved_successfully = False
        try:
            logging.debug(f"Attempt 1: Saving initial excerpt to Parquet: {abs_output_path}")
            clipped_gdf.to_parquet(abs_output_path, index=False)
            logging.debug(f"Successfully saved to Parquet on attempt 1: {abs_output_path}")
            parquet_saved_successfully = True
        except pyarrow.lib.ArrowInvalid as e_arrow_invalid:
            logging.warning(f"Attempt 1 Parquet save failed due to ArrowInvalid (likely type issue): {e_arrow_invalid}")
            logging.debug("Attempting to convert all object columns to string and retry Parquet save.")
            
            # Make a copy to avoid SettingWithCopyWarning if clipped_gdf is a slice
            clipped_gdf_copy = clipped_gdf.copy()
            
            converted_cols = []
            for col in clipped_gdf_copy.columns:
                if clipped_gdf_copy[col].dtype == 'object':
                    try:
                        clipped_gdf_copy[col] = clipped_gdf_copy[col].astype(str)
                        converted_cols.append(col)
                    except Exception as e_conv:
                        logging.warning(f"Could not convert object column '{col}' to string: {e_conv}")
            
            if converted_cols:
                logging.debug(f"Converted the following object columns to string: {', '.join(converted_cols)}")
            else:
                logging.debug("No object columns found or converted. Retrying Parquet save with original types.")

            try:
                logging.debug(f"Attempt 2: Saving excerpt to Parquet after type conversion: {abs_output_path}")
                clipped_gdf_copy.to_parquet(abs_output_path, index=False) # Use the copy
                logging.debug(f"Successfully saved to Parquet on attempt 2: {abs_output_path}")
                parquet_saved_successfully = True
                clipped_gdf = clipped_gdf_copy # Update original gdf if second attempt was successful
            except Exception as e_parquet_retry:
                logging.error(f"Attempt 2 Parquet save also failed: {e_parquet_retry}", exc_info=True)
        except Exception as e_parquet_initial:
            logging.error(f"Initial Parquet save failed with a non-ArrowInvalid error: {e_parquet_initial}", exc_info=True)

        if not parquet_saved_successfully:
            logging.info("Parquet save failed. Attempting to save as GeoPackage as a fallback.")
            gpkg_filename = os.path.splitext(output_filename)[0] + ".gpkg"
            abs_gpkg_output_path = os.path.join(abs_output_dir, gpkg_filename) 
            try:
                clipped_gdf.to_file(abs_gpkg_output_path, driver="GPKG", layer=os.path.splitext(output_filename)[0])
                logging.info(f"Successfully saved excerpt as GeoPackage: {abs_gpkg_output_path}")
                abs_output_path = abs_gpkg_output_path # Update path for subsequent operations
            except Exception as e_gpkg:
                logging.error(f"Also failed to save as GeoPackage: {e_gpkg}", exc_info=True)
                return False # Critical failure if even GeoPackage doesn't work
        # --- End Robust Parquet Saving ---

        file_size_mb = os.path.getsize(abs_output_path) / (1024 * 1024)
        logging.info(f"Final excerpt file size: {file_size_mb:.2f} MB for file {abs_output_path}.")
        
        return True

    except Exception as e:
        logging.error(f"Health Monitoring excerpt creation process failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create vector excerpts for Health Monitoring data.")
    parser.add_argument(
        '--bbox-name',
        type=str,
        default='les landes',
        choices=EXCERPT_BOUNDING_BOXES.keys(),
        help=f"The name of the bounding box to use for the excerpt. "
             f"Defined in src/config/constants.py. Choices: {list(EXCERPT_BOUNDING_BOXES.keys())}"
    )
    args = parser.parse_args()

    if main(args.bbox_name):
        logging.debug("Health Monitoring excerpt creation process finished successfully.")
    else:
        logging.error("Health Monitoring excerpt creation process failed or produced no output.") 