"""
Script to perform disturbance attribution using preprocessed data.
"""
import logging
from pathlib import Path
import sys
import geopandas as gpd
import pandas as pd # Added for pd.concat if needed, and generally useful
import time
import argparse # Import argparse
import shutil # Import shutil for clearing cache directory
import warnings # Import warnings
import subprocess # For calling the QC script

# Suppress specific FutureWarning from sklearn used by hdbscan
warnings.simplefilter(action='ignore', category=FutureWarning) 

# Add src to Python path to allow direct execution of the script
# and for imports to work correctly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Assuming Attribution class is in src.attribution.pipeline
# Based on the AGENTS.md: "A new src/attribution/ package implements the updated graph-based pipeline described in ATTRIBUTION.md"
# And the provided src/attribution/ contents show pipeline.py
from src.attribution.pipeline import Attribution, AttributionParams
# Import new constants for final aggregation
from src.config.constants import FINAL_TARGET_CLASSES

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PREPROCESSED_DIR = BASE_DIR / "outputs" / "preprocessing"
OUTPUT_ATTRIBUTION_DIR = BASE_DIR / "outputs" / "attribution"

# Define the expected preprocessed files
# These should match the output names from preprocess_excerpts.py
PREPROCESSED_FILES = {
    "cdi": "cdi_processed.parquet",
    "firepolygons": "firepolygons_processed.parquet",
    "hm": "hm_processed.parquet",
    "forms": "forms_processed.parquet",
    "senfseidl": "senfseidl_processed.parquet",
}

# --- Define and create a temporary directory for attribution caching ---
TEMP_ATTRIBUTION_DIR = Path("outputs/temp_attribution")
TEMP_ATTRIBUTION_DIR.mkdir(parents=True, exist_ok=True)
# --- End temp dir setup ---

def load_preprocessed_data(data_dir: Path, files_dict: dict) -> dict[str, gpd.GeoDataFrame]:
    """Load preprocessed GeoDataFrames from parquet files."""
    gdfs = {}
    logging.info(f"Loading preprocessed data from: {data_dir}")
    for key, filename in files_dict.items():
        file_path = data_dir / filename
        if file_path.exists():
            logging.info(f"Loading {key} from {file_path}...")
            try:
                gdfs[key] = gpd.read_parquet(file_path)
                logging.info(f"Successfully loaded {key}, {len(gdfs[key])} records.")
            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}", exc_info=True)
                # Decide if you want to raise an error or continue without this dataset
                # For now, we'll log and continue, Attribution class might handle missing keys
        else:
            logging.warning(f"Preprocessed file not found for {key}: {file_path}")
    return gdfs

def main():
    """
    Main function to run the disturbance attribution pipeline.
    """
    parser = argparse.ArgumentParser(description="Run the disturbance attribution pipeline.")
    parser.add_argument("--recompute-prepared-data", action="store_true", help="Force recomputation of prepared data, ignoring cache.")
    parser.add_argument("--recompute-graph", action="store_true", help="Force rebuilding of the graph, ignoring cache.")
    parser.add_argument("--recompute-communities", action="store_true", help="Force redetection of communities, ignoring cache.")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the temporary attribution cache directory before running.")
    args = parser.parse_args()

    if args.clear_cache:
        if TEMP_ATTRIBUTION_DIR.exists():
            try:
                shutil.rmtree(TEMP_ATTRIBUTION_DIR)
                logging.info(f"Successfully cleared cache directory: {TEMP_ATTRIBUTION_DIR}")
            except Exception as e:
                logging.error(f"Could not clear cache directory {TEMP_ATTRIBUTION_DIR}: {e}")
        else:
            logging.info(f"Cache directory {TEMP_ATTRIBUTION_DIR} does not exist. Nothing to clear.")
        # Recreate the directory after clearing
        TEMP_ATTRIBUTION_DIR.mkdir(parents=True, exist_ok=True)

    # --- ASCII Art Welcome Message ---
    print("""
    *************************************************
    *                                               *
    *        Forest Disturbance Attribution         *
    *                   Pipeline                    *
    *                                               *
    *            /\             /\                  *
    *           /\*\           /\*\                 *
    *          /\O\*\         /\O\*\                *
    *         /\*\O\*\       /\*\O\*\               *
    *        /\O\*\O\*\     /\O\*\O\*\              *
    *       /\*\O\*\O\*\   /\*\O\*\O\*\             *
    *      /\O\*\O\*\O\*\ /\O\*\O\*\O\*\            *
    *            ||             ||                  *
    *            ||             ||                  *
    *                                               *
    *************************************************
    """)
    # ---------------------------------

    logging.info("Starting disturbance attribution pipeline.")

    # Ensure output directory exists
    OUTPUT_ATTRIBUTION_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Attribution output directory created/ensured: {OUTPUT_ATTRIBUTION_DIR}")

    # 1. Load preprocessed data
    preprocessed_gdfs = load_preprocessed_data(PREPROCESSED_DIR, PREPROCESSED_FILES)

    if not preprocessed_gdfs:
        logging.error("No preprocessed data loaded. Aborting attribution pipeline.")
        return
    
    logging.info(f"Loaded {len(preprocessed_gdfs)} datasets for attribution: {list(preprocessed_gdfs.keys())}")

    # 2. Instantiate Attribution class
    # The Attribution class might require all datasets to be present, 
    # or it might be robust to missing ones. This depends on its implementation.
    # Based on test_attribution.py, it seems to expect a dict of gdfs.
    try:
        logging.info("Initializing Attribution class...")
        attribution_params = AttributionParams() # Using defaults or previously adjusted values
        
        attr = Attribution(
            gdf_dict=preprocessed_gdfs,
            params=attribution_params, 
            reliability=None, # Use default reliability
            temp_dir=TEMP_ATTRIBUTION_DIR, # Pass the temporary directory
            force_recompute_prepared_data=args.recompute_prepared_data,
            force_rebuild_graph=args.recompute_graph,
            force_redetect_communities=args.recompute_communities
        )
        attr.log_parameters()
        logging.info("Attribution class initialized.")
    except Exception as e:
        logging.error(f"Error initializing Attribution class: {e}", exc_info=True)
        return

    # 3. Run the attribution steps
    timings = {}

    try:
        logging.info("Building graph...")
        start_time = time.perf_counter()
        attr.build_graph() # This method might take parameters if defaults are not suitable
        timings["build_graph"] = time.perf_counter() - start_time
        logging.info(f"Graph built in {timings['build_graph']:.2f} seconds. Graph has {attr.graph.number_of_nodes()} nodes and {attr.graph.number_of_edges()} edges.")

        logging.info("Detecting communities...")
        start_time = time.perf_counter()
        attr.detect_communities() # This method might take parameters
        timings["detect_communities"] = time.perf_counter() - start_time
        # Log some info about communities if available, e.g., number of communities
        if 'community_id' in attr.data.columns:
            num_communities = attr.data['community_id'].nunique()
            # NaN communities (nodes not assigned) are not counted by nunique() by default if dropna=True (default)
            # If you want to count NaN as a separate group, you'd use dropna=False
            logging.info(f"Communities detected in {timings['detect_communities']:.2f} seconds. Found {num_communities} distinct communities.")

            # --- Call QC script for community size distribution ---
            temp_community_data_path = TEMP_ATTRIBUTION_DIR / "data_with_communities_for_qc.parquet"
            try:
                logging.info(f"Saving data with communities for QC to: {temp_community_data_path}")
                attr.data.to_parquet(temp_community_data_path)
                
                qc_script_path = PROJECT_ROOT / "src" / "qc" / "plot_community_diagnostics.py"
                output_qc_dir = BASE_DIR / "outputs" / "qc" 
                # Ensure the main QC output directory exists for the script to use
                output_qc_dir.mkdir(parents=True, exist_ok=True)

                logging.info(f"Running community diagnostics script: {qc_script_path}")
                subprocess.run(
                    [sys.executable, str(qc_script_path), str(temp_community_data_path), "--output-dir", str(output_qc_dir)],
                    check=True,
                    capture_output=True, # Capture output
                    text=True # Decode output as text
                )
                logging.info("Community diagnostics script finished successfully.")
                # Optionally, delete the temporary file after use
                # temp_community_data_path.unlink(missing_ok=True) 
            except FileNotFoundError:
                logging.error(f"QC script not found at {qc_script_path}. Skipping community diagnostics plot.")
            except subprocess.CalledProcessError as e:
                logging.error(f"Community diagnostics script failed with error:\\nStdout:\\n{e.stdout}\\nStderr:\\n{e.stderr}")
            except Exception as e:
                logging.error(f"An error occurred while generating community diagnostics: {e}", exc_info=True)
            # --- End QC script call ---
        else:
            logging.warning("Community detection did not add 'community_id' column as expected. Skipping community diagnostics.")


        logging.info("Running HDBSCAN...")
        start_time = time.perf_counter()
        attr.run_hdbscan() # This method might take parameters
        timings["run_hdbscan"] = time.perf_counter() - start_time
        # Log some info about hdbscan results if available
        if 'hdb_id' in attr.data.columns: # Check for 'hdb_id'
            # Calculate unique clusters, excluding -1 (noise) if you want to count actual clusters
            num_clusters = attr.data['hdb_id'][attr.data['hdb_id'] != -1].nunique()
            num_noise_points = (attr.data['hdb_id'] == -1).sum()
            logging.info(f"HDBSCAN finished in {timings['run_hdbscan']:.2f} seconds. Found {num_clusters} actual clusters and {num_noise_points} noise points.")
        else:
            logging.warning("HDBSCAN did not add 'hdb_id' column as expected.")

        logging.info("Performing final attribution...")
        start_time = time.perf_counter()
        # The attribute() method in test_attribution.py returns a dict. 
        # Let's see what it contains and how to save it.
        # The Attribution class in `pipeline.py` has `data` attribute which is the final gdf. 
        # The `attribute()` method in `pipeline.py` returns self.data. 
        final_attributed_gdf = attr.attribute()
        timings["attribute"] = time.perf_counter() - start_time
        logging.info(f"Final attribution performed in {timings['attribute']:.2f} seconds.")

        # 4. Save results
        if final_attributed_gdf is not None and not final_attributed_gdf.empty:
            output_path = OUTPUT_ATTRIBUTION_DIR / "final_attributed_disturbances.parquet"
            logging.info(f"Saving final attributed GeoDataFrame to {output_path}...")
            final_attributed_gdf.to_parquet(output_path)
            logging.info(f"Successfully saved final results with {len(final_attributed_gdf)} records.")

            # --- Add stats on final attribution for Senf&Seidl events ---
            senf_events = final_attributed_gdf[final_attributed_gdf["dataset"] == "senfseidl"].copy() # Ensure it's a copy
            if not senf_events.empty:
                # Probabilities are now directly for FINAL TARGET CLASSES
                # Column names will be like prob_Fire, prob_Biotic, etc. if pipeline.py correctly uses these from constants
                # For now, assume pipeline.py produces prob_CLASSNAME where CLASSNAME matches FINAL_TARGET_CLASSES values.
                # We need to ensure the `attribute` method in pipeline.py creates columns like `prob_Fire`, `prob_Biotic`, etc.
                # based on the FINAL_TARGET_CLASSES available in self.data['class'].unique()
                
                # The `attribute` method in pipeline.py initializes prob_ columns based on self.data["class"].unique().
                # Since preprocessing scripts now put FINAL_TARGET_CLASSES values into self.data["class"],
                # the prob_ columns from attribute() should already be like prob_Fire, prob_Biotic.
                final_prob_cols = [col for col in senf_events.columns if col.startswith("prob_")]
                
                if final_prob_cols:
                    # No aggregation step needed anymore.
                    # The prob_ columns from attr.attribute() are now the final target probabilities.
                    logging.info(f"Using final probability columns directly from attribution output: {final_prob_cols}")

                    # Determine winning class and confidence based on these direct final probabilities
                    senf_events["winning_class"] = senf_events[final_prob_cols].idxmax(axis=1).str.replace("prob_", "")
                    senf_events["winning_confidence"] = senf_events[final_prob_cols].max(axis=1)

                    logging.info("--- Final Attribution Statistics for Senf&Seidl Events (Based on Direct Final Classes) ---")
                    logging.info(f"Total Senf&Seidl events processed for attribution: {len(senf_events)}")
                    
                    logging.info("Distribution of winning classes:")
                    winning_class_counts = senf_events["winning_class"].value_counts()
                    for cls, count in winning_class_counts.items():
                        logging.info(f"  {cls}: {count} events ({count / len(senf_events) * 100:.2f}%)")

                    logging.info("Confidence of winning class:")
                    logging.info(f"  Min confidence: {senf_events['winning_confidence'].min():.3f}")
                    logging.info(f"  Mean confidence: {senf_events['winning_confidence'].mean():.3f}")
                    logging.info(f"  Median confidence: {senf_events['winning_confidence'].median():.3f}")
                    logging.info(f"  Max confidence: {senf_events['winning_confidence'].max():.3f}")
                    
                    low_confidence_threshold = 0.5
                    num_low_confidence = (senf_events['winning_confidence'] <= low_confidence_threshold).sum()
                    logging.info(f"Number of Senf&Seidl events with winning confidence <= {low_confidence_threshold}: {num_low_confidence} ({num_low_confidence / len(senf_events) * 100:.2f}%)")
                    logging.info("-----------------------------------------------------------")
                else:
                    logging.warning("No probability columns (prob_*) found for Senf&Seidl events. Cannot compute final attribution stats.")
            else:
                logging.info("No Senf&Seidl events found in the final attributed data to generate stats for.")
            # --- End stats on final attribution ---

        else:
            logging.warning("Final attributed GeoDataFrame is None or empty. Nothing to save.")

        logging.info(f"Attribution pipeline completed. Timings: {timings}")

    except Exception as e:
        logging.error(f"Error during attribution steps: {e}", exc_info=True)

if __name__ == "__main__":
    main() 