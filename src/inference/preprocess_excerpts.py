"""
Script to preprocess all raw data excerpts.
"""
import logging
import os
from pathlib import Path
import sys
import geopandas as gpd
import argparse

# Add src to Python path to allow direct execution of the script
# and for imports to work correctly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import cdi, firepolygons, forms, hm, senfseidl
# Import the QC function
from src.qc.check_preprocessing import generate_qc_report

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define base paths
# Assumes the script is run from the root of the repository
# or that paths are adjusted accordingly.
# For robustness, consider making these configurable (e.g., via command-line args or a config file)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
EXCERPTS_RAW_DIR = BASE_DIR / "excerpts" / "raw"
OUTPUT_PREPROCESSING_DIR = BASE_DIR / "outputs" / "preprocessing"


def main(start_year_arg: int, end_year_arg: int):
    """
    Main function to orchestrate the preprocessing of all datasets.
    Applies a date filter based on provided start and end years.
    """
    logging.info(f"Starting preprocessing of all raw excerpts. Filtering for years: {start_year_arg}-{end_year_arg}")

    # Ensure output directory exists
    OUTPUT_PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory created/ensured: {OUTPUT_PREPROCESSING_DIR}")

    # --- Senf & Seidl ---
    logging.info("--- Starting Senf & Seidl preprocessing ---")
    try:
        senfseidl_cause_raster_path = EXCERPTS_RAW_DIR / "excerpt_fire_wind_barkbeetle_france.tif"
        senfseidl_year_raster_path = EXCERPTS_RAW_DIR / "excerpt_disturbance_year_1986-2020_france.tif"
        
        if not senfseidl_cause_raster_path.exists():
            logging.warning(f"Senf & Seidl cause raster not found at: {senfseidl_cause_raster_path}")
        if not senfseidl_year_raster_path.exists():
            logging.warning(f"Senf & Seidl year raster not found at: {senfseidl_year_raster_path}")

        if senfseidl_cause_raster_path.exists() and senfseidl_year_raster_path.exists():
            output_senfseidl_path = OUTPUT_PREPROCESSING_DIR / "senfseidl_processed.parquet"
            logging.info(f"Processing Senf & Seidl. Output: {output_senfseidl_path}")
            processed_senfseidl_gdf = senfseidl.process_senfseidl(
                attribution_raster_path=str(senfseidl_cause_raster_path),
                year_raster_path=str(senfseidl_year_raster_path),
                output_file=str(output_senfseidl_path),
                start_year=start_year_arg,
                end_year=end_year_arg
            )
            logging.info(f"Senf & Seidl preprocessing finished. Generated {len(processed_senfseidl_gdf)} records.")
            # Add QC step
            if not processed_senfseidl_gdf.empty:
                generate_qc_report(str(output_senfseidl_path), "senfseidl")
                # Log class distribution
                if 'class' in processed_senfseidl_gdf.columns and not processed_senfseidl_gdf['class'].empty:
                    logging.info("Senf & Seidl - Final Class Distribution (by unique class combinations):")
                    class_display_series = processed_senfseidl_gdf['class'].apply(
                        lambda x: ', '.join(sorted(x)) if isinstance(x, list) and x else str(x) if x is not None else 'N/A'
                    )
                    class_counts = class_display_series.value_counts()
                    total_records = len(processed_senfseidl_gdf)
                    for cls_str, count in class_counts.items():
                        logging.info(f"  - {cls_str}: {count} records ({count/total_records*100:.2f}%)")
                else:
                    logging.warning("'class' column not found or empty in processed Senf & Seidl GDF for QC stats.")
        else:
            logging.warning("Skipping Senf & Seidl preprocessing due to missing input files.")

    except Exception as e:
        logging.error(f"Error during Senf & Seidl preprocessing: {e}", exc_info=True)
    logging.info("--- Finished Senf & Seidl preprocessing ---")


    # --- Health Monitoring (HM) ---
    logging.info("--- Starting Health Monitoring (HM) preprocessing ---")
    try:
        # The excerpt is already a parquet file, as per list_dir and AGENTS.md.
        # The original hm.py expects a parquet file.
        hm_input_path = EXCERPTS_RAW_DIR / "excerpt_health_monitoring.parquet"

        if hm_input_path.exists():
            output_hm_path = OUTPUT_PREPROCESSING_DIR / "hm_processed.parquet"
            logging.info(f"Processing Health Monitoring. Input: {hm_input_path}, Output: {output_hm_path}")
            # Log count before processing for HM, as it's already vector
            try:
                raw_hm_gdf = gpd.read_parquet(hm_input_path)
                logging.info(f"Health Monitoring raw excerpt has {len(raw_hm_gdf)} records before processing.")
            except Exception as e:
                logging.warning(f"Could not read raw HM GDF for pre-count: {e}")
            
            processed_hm_gdf = hm.process_hm(
                parquet_path=str(hm_input_path),
                output_file=str(output_hm_path),
                start_year=start_year_arg,
                end_year=end_year_arg
            )
            logging.info(f"Health Monitoring preprocessing finished. Generated {len(processed_hm_gdf)} records.")
            # Add QC step
            if not processed_hm_gdf.empty:
                generate_qc_report(str(output_hm_path), "hm")
                # Log class distribution
                if 'class' in processed_hm_gdf.columns and not processed_hm_gdf['class'].empty:
                    logging.info("Health Monitoring - Final Class Distribution (by unique class combinations):")
                    class_display_series = processed_hm_gdf['class'].apply(
                        lambda x: ', '.join(sorted(x)) if isinstance(x, list) and x else str(x) if x is not None else 'N/A'
                    )
                    class_counts = class_display_series.value_counts()
                    total_records = len(processed_hm_gdf)
                    for cls_str, count in class_counts.items():
                        logging.info(f"  - {cls_str}: {count} records ({count/total_records*100:.2f}%)")
                else:
                    logging.warning("'class' column not found or empty in processed HM GDF for QC stats.")
        else:
            logging.warning(f"Skipping Health Monitoring preprocessing, input file not found: {hm_input_path}")

    except Exception as e:
        logging.error(f"Error during Health Monitoring preprocessing: {e}", exc_info=True)
    logging.info("--- Finished Health Monitoring preprocessing ---")


    # --- Fire Polygons ---
    logging.info("--- Starting Fire Polygons preprocessing ---")
    try:
        firepolygons_gpkg_dir = EXCERPTS_RAW_DIR / "firepolygons_gpkg"
        firepolygons_csv_path = EXCERPTS_RAW_DIR / "FFUD_Inventory_Arthur_excerpt.csv"

        if not firepolygons_gpkg_dir.is_dir():
             logging.warning(f"Fire Polygons GPKG directory not found or not a directory: {firepolygons_gpkg_dir}")
        if not firepolygons_csv_path.exists():
            logging.warning(f"Fire Polygons CSV attributes file not found: {firepolygons_csv_path}")

        if firepolygons_gpkg_dir.is_dir() and firepolygons_csv_path.exists():
            output_firepolygons_path = OUTPUT_PREPROCESSING_DIR / "firepolygons_processed.parquet"
            logging.info(f"Processing Fire Polygons. GPKG Dir: {firepolygons_gpkg_dir}, CSV: {firepolygons_csv_path}, Output: {output_firepolygons_path}")
            processed_firepolygons_gdf = firepolygons.process_firepolygons(
                csv_file=str(firepolygons_csv_path),
                polygon_dir=str(firepolygons_gpkg_dir),
                output_file=str(output_firepolygons_path),
                start_year=start_year_arg,
                end_year=end_year_arg
            )
            logging.info(f"Fire Polygons preprocessing finished. Generated {len(processed_firepolygons_gdf)} records.")
            # Add QC step
            if not processed_firepolygons_gdf.empty:
                generate_qc_report(str(output_firepolygons_path), "firepolygons")
                # Log class distribution
                if 'class' in processed_firepolygons_gdf.columns and not processed_firepolygons_gdf['class'].empty:
                    logging.info("Fire Polygons - Final Class Distribution (by unique class combinations):")
                    class_display_series = processed_firepolygons_gdf['class'].apply(
                        lambda x: ', '.join(sorted(x)) if isinstance(x, list) and x else str(x) if x is not None else 'N/A'
                    )
                    class_counts = class_display_series.value_counts()
                    total_records = len(processed_firepolygons_gdf)
                    for cls_str, count in class_counts.items():
                        logging.info(f"  - {cls_str}: {count} records ({count/total_records*100:.2f}%)")
                else:
                    logging.warning("'class' column not found or empty in processed Fire Polygons GDF for QC stats.")
        else:
            logging.warning(f"Skipping Fire Polygons preprocessing due to missing inputs.")

    except Exception as e:
        logging.error(f"Error during Fire Polygons preprocessing: {e}", exc_info=True)
    logging.info("--- Finished Fire Polygons preprocessing ---")


    # --- Combined Drought Indicator (CDI) ---
    logging.info("--- Starting CDI preprocessing ---")
    try:
        cdi_rasters_dir = EXCERPTS_RAW_DIR / "cdi"

        if not cdi_rasters_dir.is_dir():
            logging.warning(f"CDI raster directory not found or not a directory: {cdi_rasters_dir}")
        
        if cdi_rasters_dir.is_dir() and any(cdi_rasters_dir.iterdir()):
            output_cdi_path = OUTPUT_PREPROCESSING_DIR / "cdi_processed.parquet"
            logging.info(f"Processing CDI. Input Dir: {cdi_rasters_dir}, Output: {output_cdi_path}")
            processed_cdi_gdf = cdi.process_cdi(
                input_dir=str(cdi_rasters_dir),
                output_file=str(output_cdi_path),
                start_year=start_year_arg,
                end_year=end_year_arg
            )
            logging.info(f"CDI preprocessing finished. Generated {len(processed_cdi_gdf)} records.")
            # Add QC step
            if not processed_cdi_gdf.empty:
                generate_qc_report(str(output_cdi_path), "cdi")
                # Log class distribution
                if 'class' in processed_cdi_gdf.columns and not processed_cdi_gdf['class'].empty:
                    logging.info("CDI - Final Class Distribution (by unique class combinations):")
                    class_display_series = processed_cdi_gdf['class'].apply(
                        lambda x: ', '.join(sorted(x)) if isinstance(x, list) and x else str(x) if x is not None else 'N/A'
                    )
                    class_counts = class_display_series.value_counts()
                    total_records = len(processed_cdi_gdf)
                    for cls_str, count in class_counts.items():
                        logging.info(f"  - {cls_str}: {count} records ({count/total_records*100:.2f}%)")
                else:
                    logging.warning("'class' column not found or empty in processed CDI GDF for QC stats.")
        else:
            logging.warning(f"Skipping CDI preprocessing, input directory not found or empty: {cdi_rasters_dir}")

    except Exception as e:
        logging.error(f"Error during CDI preprocessing: {e}", exc_info=True)
    logging.info("--- Finished CDI preprocessing ---")


    # --- FORMS (Forest Height) ---
    logging.info("--- Starting FORMS preprocessing ---")
    try:
        # Using the two year-specific rasters for difference calculation
        forms_raster_path_2022 = EXCERPTS_RAW_DIR / "excerpt_Height_mavg_2022.tif"
        forms_raster_path_2023 = EXCERPTS_RAW_DIR / "excerpt_Height_mavg_2023.tif"
        forms_raster_path_2018 = EXCERPTS_RAW_DIR / "excerpt_Height_mavg_2018.tif"
        forms_raster_path_2019 = EXCERPTS_RAW_DIR / "excerpt_Height_mavg_2019.tif"

        
        # The forms.process_forms function expects a list of raster paths, sorted by date.
        forms_raster_paths = []
        if forms_raster_path_2022.exists():
            forms_raster_paths.append(str(forms_raster_path_2022))
        else:
            logging.warning(f"FORMS raster 2022 not found: {forms_raster_path_2022}")
            
        if forms_raster_path_2023.exists():
            forms_raster_paths.append(str(forms_raster_path_2023))
        else:
            logging.warning(f"FORMS raster 2023 not found: {forms_raster_path_2023}")

        if forms_raster_path_2018.exists():
            forms_raster_paths.append(str(forms_raster_path_2018))
        else:
            logging.warning(f"FORMS raster 2018 not found: {forms_raster_path_2018}")
            
        if forms_raster_path_2019.exists():
            forms_raster_paths.append(str(forms_raster_path_2019))
        else:
            logging.warning(f"FORMS raster 2019 not found: {forms_raster_path_2019}")

        # Ensure there are at least two rasters to compute differences
        if len(forms_raster_paths) >= 2:
            output_forms_path = OUTPUT_PREPROCESSING_DIR / "forms_processed.parquet"
            logging.info(f"Processing FORMS. Input Rasters: {forms_raster_paths}, Output: {output_forms_path}")
            processed_forms_gdf = forms.process_forms(
                rasters=forms_raster_paths, # Ensure this is a list of strings
                output_file=str(output_forms_path),
                start_year_filter=start_year_arg,
                end_year_filter=end_year_arg
            )
            logging.info(f"FORMS preprocessing finished. Generated {len(processed_forms_gdf)} records.")
            # Add QC step
            if not processed_forms_gdf.empty:
                generate_qc_report(str(output_forms_path), "forms")
                # Log class distribution
                if 'class' in processed_forms_gdf.columns and not processed_forms_gdf['class'].empty:
                    logging.info("FORMS - Final Class Distribution (by unique class combinations):")
                    class_display_series = processed_forms_gdf['class'].apply(
                        lambda x: ', '.join(sorted(x)) if isinstance(x, list) and x else str(x) if x is not None else 'N/A'
                    )
                    class_counts = class_display_series.value_counts()
                    total_records = len(processed_forms_gdf)
                    for cls_str, count in class_counts.items():
                        logging.info(f"  - {cls_str}: {count} records ({count/total_records*100:.2f}%)")
                else:
                    logging.warning("'class' column not found or empty in processed FORMS GDF for QC stats.")
        else:
            logging.warning(f"Skipping FORMS preprocessing, not enough rasters found for difference calculation (need at least 2). Found: {len(forms_raster_paths)}")

    except Exception as e:
        logging.error(f"Error during FORMS preprocessing: {e}", exc_info=True)
    logging.info("--- Finished FORMS preprocessing ---")

    logging.info("All preprocessing tasks attempted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw data excerpts with date filtering.")
    parser.add_argument(
        "--start-year", 
        type=int, 
        default=2016, 
        help="Start year for filtering data (inclusive). Default: 2016."
    )
    parser.add_argument(
        "--end-year", 
        type=int, 
        default=2021, 
        help="End year for filtering data (inclusive). Default: 2021."
    )
    args = parser.parse_args()

    if args.start_year > args.end_year:
        logging.error("Start year cannot be after end year.")
        sys.exit(1)

    main(args.start_year, args.end_year) 