import os
import logging
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import numpy as np
import pyarrow # Added for specific error catching

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


# BBOX coordinates (minx, miny, maxx, maxy) in EPSG:2154
BBOX_COORDS_EPSG2154 = (307783.0822, 6340505.4366, 469246.8845, 6419190.9011)
TARGET_CRS_EPSG2154 = "EPSG:2154"
INITIAL_DATA_CRS_EPSG4326 = "EPSG:4326"

# Output directory and filename, relative to this script's location
OUTPUT_DIR_RELATIVE = "../excerpts/raw/"
OUTPUT_FILENAME = "excerpt_health_monitoring.parquet"

# File size limits in MB
MAX_SIZE_MB_WARN_INITIAL = 5.0
TARGET_SIZE_MB_HARD_LIMIT = 10.0 # Max desired size after sampling
TARGET_SIZE_MB_BEFORE_FINAL_CHECK = 9.5 # Target for sampling ratio to have a small margin

def create_health_monitoring_excerpt():
    """
    Reads the Health Monitoring Excel file, creates a GeoDataFrame,
    clips it to the BBOX, handles problematic data types, and saves it as Parquet.
    Includes file size management and robust Parquet saving with GeoPackage fallback.
    """
    logging.info("Starting Health Monitoring data excerpt creation process...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_output_dir = os.path.abspath(os.path.join(script_dir, OUTPUT_DIR_RELATIVE))
    os.makedirs(abs_output_dir, exist_ok=True)
    abs_output_path = os.path.join(abs_output_dir, OUTPUT_FILENAME)

    if not os.path.exists(EXCEL_FILE_PATH):
        logging.error(f"Source Excel file not found: {EXCEL_FILE_PATH}. Aborting.")
        return False

    try:
        logging.info(f"Attempting to load Excel file: {EXCEL_FILE_PATH}, sheet: {EXCEL_SHEET_NAME}, header row: {EXCEL_HEADER_ROW}")
        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=EXCEL_SHEET_NAME, header=EXCEL_HEADER_ROW)
        logging.info(f"Successfully loaded Excel data. {len(df)} rows found.")

        # Data Cleaning & Preparation
        df.dropna(subset=[LON_COL, LAT_COL], inplace=True)
        logging.info(f"{len(df)} rows remaining after dropping NA in lat/lon.")
        
        # Initial explicit conversion for known problematic columns
        for col, dtype in PROBLEM_COLUMN_HANDLING.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                    logging.info(f"Successfully converted column '{col}' to {dtype}.")
                except Exception as e:
                    logging.warning(f"Could not convert column '{col}' to {dtype}: {e}. Skipping conversion for this column.")
            else:
                logging.warning(f"Problematic column '{col}' configured for handling not found in DataFrame.")


        geometry = [Point(xy) for xy in zip(df[LON_COL], df[LAT_COL])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=INITIAL_DATA_CRS_EPSG4326)
        logging.info(f"Successfully converted to GeoDataFrame with initial CRS: {INITIAL_DATA_CRS_EPSG4326}")

        logging.info(f"Reprojecting GeoDataFrame from {INITIAL_DATA_CRS_EPSG4326} to {TARGET_CRS_EPSG2154}.")
        gdf = gdf.to_crs(TARGET_CRS_EPSG2154)

        logging.info(f"Clipping GeoDataFrame to BBOX in {TARGET_CRS_EPSG2154}.")
        bbox_polygon = box(*BBOX_COORDS_EPSG2154)
        clipped_gdf = gdf[gdf.geometry.intersects(bbox_polygon)].copy()
        
        if clipped_gdf.empty:
            logging.warning("No features found within the BBOX. Output file will not be created.")
            return False
        
        logging.info(f"{len(clipped_gdf)} features found within BBOX.")

        # --- Robust Parquet Saving --- 
        parquet_saved_successfully = False
        try:
            logging.info(f"Attempt 1: Saving initial excerpt to Parquet: {abs_output_path}")
            clipped_gdf.to_parquet(abs_output_path, index=False)
            logging.info(f"Successfully saved to Parquet on attempt 1: {abs_output_path}")
            parquet_saved_successfully = True
        except pyarrow.lib.ArrowInvalid as e_arrow_invalid:
            logging.warning(f"Attempt 1 Parquet save failed due to ArrowInvalid (likely type issue): {e_arrow_invalid}")
            logging.info("Attempting to convert all object columns to string and retry Parquet save.")
            
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
                logging.info(f"Converted the following object columns to string: {', '.join(converted_cols)}")
            else:
                logging.info("No object columns found or converted. Retrying Parquet save with original types.")

            try:
                logging.info(f"Attempt 2: Saving excerpt to Parquet after type conversion: {abs_output_path}")
                clipped_gdf_copy.to_parquet(abs_output_path, index=False) # Use the copy
                logging.info(f"Successfully saved to Parquet on attempt 2: {abs_output_path}")
                parquet_saved_successfully = True
                clipped_gdf = clipped_gdf_copy # Update original gdf if second attempt was successful
            except Exception as e_parquet_retry:
                logging.error(f"Attempt 2 Parquet save also failed: {e_parquet_retry}", exc_info=True)
        except Exception as e_parquet_initial:
            logging.error(f"Initial Parquet save failed with a non-ArrowInvalid error: {e_parquet_initial}", exc_info=True)

        if not parquet_saved_successfully:
            logging.info("Parquet save failed. Attempting to save as GeoPackage as a fallback.")
            gpkg_filename = os.path.splitext(OUTPUT_FILENAME)[0] + ".gpkg"
            abs_gpkg_output_path = os.path.join(abs_output_dir, gpkg_filename) 
            try:
                clipped_gdf.to_file(abs_gpkg_output_path, driver="GPKG", layer=os.path.splitext(OUTPUT_FILENAME)[0])
                logging.info(f"Successfully saved excerpt as GeoPackage: {abs_gpkg_output_path}")
                abs_output_path = abs_gpkg_output_path # Update path for size check and subsequent operations
            except Exception as e_gpkg:
                logging.error(f"Also failed to save as GeoPackage: {e_gpkg}", exc_info=True)
                return False # Critical failure if even GeoPackage doesn't work
        # --- End Robust Parquet Saving ---

        # File size management (operates on abs_output_path, which is now either .parquet or .gpkg)
        initial_size_bytes = os.path.getsize(abs_output_path)
        initial_size_mb = initial_size_bytes / (1024 * 1024)
        logging.info(f"Initial excerpt file size: {initial_size_mb:.2f} MB for file {abs_output_path}.")

        if initial_size_mb > MAX_SIZE_MB_WARN_INITIAL:
            logging.warning(f"Initial excerpt size ({initial_size_mb:.2f} MB) is > {MAX_SIZE_MB_WARN_INITIAL}MB.")
            if initial_size_mb > TARGET_SIZE_MB_HARD_LIMIT:
                sampling_ratio = TARGET_SIZE_MB_BEFORE_FINAL_CHECK / initial_size_mb
                logging.warning(
                    f"Initial excerpt ({initial_size_mb:.2f} MB) > {TARGET_SIZE_MB_HARD_LIMIT}MB. "
                    f"Attempting to sample {sampling_ratio*100:.2f}% of rows to reduce size."
                )
                
                sampled_gdf = clipped_gdf.sample(frac=sampling_ratio, random_state=42)
                
                try:
                    if abs_output_path.endswith(".parquet"):
                        sampled_gdf.to_parquet(abs_output_path, index=False)
                    elif abs_output_path.endswith(".gpkg"):
                         sampled_gdf.to_file(abs_output_path, driver="GPKG", layer=os.path.splitext(OUTPUT_FILENAME)[0])
                    logging.info(f"Resampled excerpt saved to {abs_output_path}.")
                except Exception as e_sample_save:
                    logging.error(f"Error saving resampled excerpt: {e_sample_save}", exc_info=True)
                    # If saving sampled data fails, we might still have the larger original (gpkg or parquet if second attempt worked)
                    # For simplicity, we don't delete it here, but log a strong warning.
                    logging.critical(f"Failed to save SAMPLED data. The larger original file {abs_output_path} might still exist.")
                    return False # Treat as failure for sampling stage

                final_size_bytes = os.path.getsize(abs_output_path)
                final_size_mb = final_size_bytes / (1024 * 1024)
                logging.info(f"Resampled excerpt size: {final_size_mb:.2f} MB.")

                if final_size_mb > TARGET_SIZE_MB_HARD_LIMIT:
                    logging.critical(
                        f"Resampled excerpt ({final_size_mb:.2f} MB) is STILL > {TARGET_SIZE_MB_HARD_LIMIT}MB. "
                        "Manual intervention might be needed."
                    )
            else:
                logging.info(f"Final excerpt size: {initial_size_mb:.2f} MB (within hard limit but triggered initial warning).")
        else:
            logging.info(f"Final excerpt size: {initial_size_mb:.2f} MB (within initial warning limit).")
        
        return True

    except Exception as e:
        logging.error(f"Health Monitoring excerpt creation process failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    if create_health_monitoring_excerpt():
        logging.info("Health Monitoring excerpt creation process finished successfully.")
    else:
        logging.error("Health Monitoring excerpt creation process failed or produced no output.") 