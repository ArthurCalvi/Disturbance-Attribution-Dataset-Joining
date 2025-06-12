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
import numpy as np # Added for np.ndarray
import json # To save/load parameters

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
    parser = argparse.ArgumentParser(
        description="Run the disturbance attribution pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Cache control arguments ---
    cache_group = parser.add_argument_group('Cache Control')
    cache_group.add_argument("--recompute-prepared-data", action="store_true", help="Force recomputation of prepared data, ignoring cache.")
    cache_group.add_argument("--recompute-graph", action="store_true", help="Force rebuilding of the graph, ignoring cache.")
    cache_group.add_argument("--recompute-communities", action="store_true", help="Force redetection of communities, ignoring cache.")
    cache_group.add_argument("--clear-cache", action="store_true", help="Clear the temporary attribution cache directory before running.")
    
    # --- Attribution parameter arguments ---
    # Defaults are taken from AttributionParams if not provided.
    param_group = parser.add_argument_group('Attribution Hyperparameters')
    param_group.add_argument('--spatial-half-life', type=float, default=None, help="Spatial decay half-life in meters.")
    param_group.add_argument('--temporal-half-life', type=float, default=None, help="Temporal decay half-life in days.")
    param_group.add_argument('--max-spatial-dist-m', type=float, default=None, help="Maximum spatial distance for candidate pairs (meters).")
    param_group.add_argument('--max-temporal-dist-days', type=float, default=None, help="Maximum temporal distance for candidate pairs (days).")
    param_group.add_argument('--lambda-intra', type=float, default=None, help="Down-weighting factor for within-dataset links.")
    param_group.add_argument('--louvain-resolution', type=float, default=None, help="Resolution parameter for Louvain algorithm.")
    param_group.add_argument('--alpha-t', type=float, default=None, help="HDBSCAN temporal scaling factor.")
    param_group.add_argument('--alpha-c', type=float, default=None, help="HDBSCAN cause penalty (spatial equivalent in meters).")
    param_group.add_argument('--hdbscan-min-cluster-size-abs', type=int, default=None, help="Absolute minimum cluster size for HDBSCAN.")
    param_group.add_argument('--hdbscan-min-cluster-size-rel', type=float, default=None, help="Relative min cluster size for HDBSCAN (fraction of community).")
    param_group.add_argument('--hdbscan-min-samples-abs', type=int, default=None, help="Absolute minimum samples for HDBSCAN.")
    param_group.add_argument('--hdbscan-min-samples-rel', type=float, default=None, help="Relative min samples for HDBSCAN (fraction of community).")
    param_group.add_argument('--senf-self-vote-factor', type=float, default=None, help="Weight factor for a Senf&Seidl polygon's self-vote.")

    args = parser.parse_args()

    # --- Parameter-based cache validation ---
    params_path = TEMP_ATTRIBUTION_DIR / "params.json"
    
    # Create AttributionParams instance based on defaults and CLI args for comparison
    cli_params_overrides = {
        key: value for key, value in vars(args).items() 
        if key in AttributionParams.__dataclass_fields__ and value is not None
    }
    current_params = AttributionParams(**cli_params_overrides)
    
    should_clear_cache_due_to_params = False

    if not args.clear_cache and params_path.exists():
        try:
            with open(params_path, 'r') as f:
                cached_params_dict = json.load(f)
            logging.info("Comparing current parameters with cached parameters...")
            
            current_params_dict = current_params.__dict__
            
            if cached_params_dict != current_params_dict:
                logging.warning("Attribution parameters have changed. Forcing cache clear.")
                # Find what changed for better logging
                changed_keys = {
                    k for k in current_params_dict 
                    if current_params_dict.get(k) != cached_params_dict.get(k)
                }
                for k in changed_keys:
                    cached_val = cached_params_dict.get(k, 'Not in cache')
                    current_val = current_params_dict.get(k)
                    logging.warning(f"  - {k}: cached='{cached_val}', current='{current_val}'")
                should_clear_cache_due_to_params = True
            else:
                logging.info("Parameters match cached version. Using existing cache.")

        except Exception as e:
            logging.warning(f"Could not read or compare cached parameters file: {e}. Forcing cache clear.")
            should_clear_cache_due_to_params = True
            
    elif not args.clear_cache and not params_path.exists() and any(TEMP_ATTRIBUTION_DIR.iterdir()):
        # If params file doesn't exist but cache dir is not empty, it's a stale cache.
        logging.warning("Cache directory contains data but no parameters file. Forcing cache clear.")
        should_clear_cache_due_to_params = True

    if args.clear_cache or should_clear_cache_due_to_params:
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

        attr = Attribution(
            gdf_dict=preprocessed_gdfs,
            params=current_params, # Use the params object created earlier
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
            
            # Analyze community composition for Senf&Seidl-only communities
            all_data_grouped_by_community = attr.data.groupby('community_id')['dataset'].apply(lambda x: set(x))
            senf_only_community_ids = all_data_grouped_by_community[all_data_grouped_by_community == {'senfseidl'}].index.tolist()
            
            if senf_only_community_ids:
                # Debugging log remains here as it was useful
                senf_events_for_debug = attr.data[
                    (attr.data['dataset'] == 'senfseidl') &
                    (attr.data['community_id'].isin(senf_only_community_ids))
                ]
                if not senf_events_for_debug.empty:
                    logging.info("--- Debugging Senf&Seidl 'class' column (sample from S&S-only communities) ---")
                    for i, (idx, row) in enumerate(senf_events_for_debug.iterrows()):
                        if i < 5:
                            class_val_debug = row.get('class')
                            raw_cause_debug = row.get('raw_cause_description', 'N/A')
                            logging.info(f"  Sample {i}: Index {idx}, RawCause: '{raw_cause_debug}', Class: {class_val_debug} (Type: {type(class_val_debug)}), IsListOrArray: {isinstance(class_val_debug, (list, np.ndarray))}, Len>1: {isinstance(class_val_debug, (list, np.ndarray)) and len(class_val_debug) > 1}")
                        else:
                            break
                    logging.info("---------------------------------------------------------------------------------")

                senf_only_data = attr.data[
                    (attr.data['community_id'].isin(senf_only_community_ids)) &
                    (attr.data['dataset'] == 'senfseidl')
                ]
                total_events_in_senf_only_communities = len(senf_only_data)
                
                ambiguous_event_count_in_senf_only_communities = 0
                for idx, row in senf_only_data.iterrows():
                    class_val = row.get('class')
                    # MODIFIED AMBIGUITY CHECK
                    if isinstance(class_val, (list, np.ndarray)) and len(class_val) > 1:
                        ambiguous_event_count_in_senf_only_communities += 1
                
                logging.info(f"--- Senf&Seidl-only Community Analysis (Pre-Attribution) ---")
                logging.info(f"Number of communities containing ONLY Senf&Seidl polygons: {len(senf_only_community_ids)}")
                logging.info(f"Total Senf&Seidl events in these Senf&Seidl-only communities: {total_events_in_senf_only_communities}")
                if total_events_in_senf_only_communities > 0:
                    percentage_ambiguous = (ambiguous_event_count_in_senf_only_communities / total_events_in_senf_only_communities) * 100
                    logging.info(f"Senf&Seidl events with ambiguous classes (list/array length > 1) in these communities: {ambiguous_event_count_in_senf_only_communities} ({percentage_ambiguous:.1f}%)")
                else:
                    logging.info(f"Senf&Seidl events with ambiguous classes (list/array length > 1) in these communities: 0 (0.0%)")
                logging.info(f"---------------------------------------------------------------")

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
            # --- Standardize final attribution class and confidence columns ---
            logging.info("Standardizing final attribution class and confidence columns for all datasets...")
            # Identify probability columns (e.g., 'prob_Fire', 'prob_Biotic')
            prob_cols = [col for col in final_attributed_gdf.columns if col.startswith("prob_") and col not in ["prob_sum", "prob_max"]] # Exclude helper columns if they exist
            if prob_cols:
                logging.info(f"Found probability columns: {prob_cols}")
                final_attributed_gdf['temp_prob_sum'] = final_attributed_gdf[prob_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
                rows_with_probs_mask = final_attributed_gdf['temp_prob_sum'] > 0
                final_attributed_gdf.loc[rows_with_probs_mask, 'attributed_class_final'] = final_attributed_gdf.loc[rows_with_probs_mask, prob_cols].idxmax(axis=1).str.replace("prob_", "")
                final_attributed_gdf.loc[rows_with_probs_mask, 'attribution_confidence_final'] = final_attributed_gdf.loc[rows_with_probs_mask, prob_cols].max(axis=1)
                logging.info(f"{rows_with_probs_mask.sum()} records had attribution derived from probability columns.")
                rows_without_probs_mask = ~rows_with_probs_mask
                if 'class' in final_attributed_gdf.columns:
                    final_attributed_gdf.loc[rows_without_probs_mask, 'attributed_class_final'] = final_attributed_gdf.loc[rows_without_probs_mask, 'class'].apply(
                        lambda x: ', '.join(sorted(x)) if isinstance(x, list) and x else (x if isinstance(x, str) else 'Unknown')
                    )
                    final_attributed_gdf.loc[rows_without_probs_mask, 'attribution_confidence_final'] = 1.0
                    logging.info(f"{rows_without_probs_mask.sum()} records had attribution derived from pre-existing 'class' column with confidence 1.0.")
                else:
                    final_attributed_gdf.loc[rows_without_probs_mask, 'attributed_class_final'] = 'Unknown'
                    final_attributed_gdf.loc[rows_without_probs_mask, 'attribution_confidence_final'] = 0.0
                    logging.warning("'class' column not found for records without probabilities. Attributed class set to 'Unknown'.")
                final_attributed_gdf.drop(columns=['temp_prob_sum'], inplace=True, errors='ignore')
            else:
                logging.warning("No probability columns (prob_*) found in the attribution output. Deriving final class from 'class' column for all records.")
                if 'class' in final_attributed_gdf.columns:
                    final_attributed_gdf['attributed_class_final'] = final_attributed_gdf['class'].apply(
                        lambda x: ', '.join(sorted(x)) if isinstance(x, list) and x else (x if isinstance(x, str) else 'Unknown')
                    )
                    final_attributed_gdf['attribution_confidence_final'] = 1.0
                else:
                    final_attributed_gdf['attributed_class_final'] = 'Unknown'
                    final_attributed_gdf['attribution_confidence_final'] = 0.0
                    logging.error("'class' column also not found. Cannot determine final attributed class.")

            # --- Log new community/isolation stats (Post-Attribution) ---
            if 'community_id' in final_attributed_gdf.columns and 'dataset' in final_attributed_gdf.columns and 'class' in final_attributed_gdf.columns:
                senf_gdf_final = final_attributed_gdf[final_attributed_gdf['dataset'] == 'senfseidl']
                
                final_all_data_grouped_by_community = final_attributed_gdf.groupby('community_id')['dataset'].apply(lambda x: set(x))
                final_senf_only_community_ids = final_all_data_grouped_by_community[final_all_data_grouped_by_community == {'senfseidl'}].index.tolist()

                num_senf_only_communities_final = len(final_senf_only_community_ids)
                events_in_senf_only_communities_final = senf_gdf_final[senf_gdf_final['community_id'].isin(final_senf_only_community_ids)]
                total_events_in_senf_only_final = len(events_in_senf_only_communities_final)
                
                ambiguous_events_in_senf_only_final_count = 0
                for _, row in events_in_senf_only_communities_final.iterrows():
                    class_val = row.get('class')
                    # MODIFIED AMBIGUITY CHECK
                    if isinstance(class_val, (list, np.ndarray)) and len(class_val) > 1:
                        ambiguous_events_in_senf_only_final_count += 1
                
                logging.info(f"--- Post-Attribution Senf&Seidl Community Stats ---")
                logging.info(f"Number of Senf&Seidl-only communities: {num_senf_only_communities_final}")
                logging.info(f"Total Senf&Seidl events in these Senf&Seidl-only communities: {total_events_in_senf_only_final}")
                if total_events_in_senf_only_final > 0:
                    perc_ambiguous_in_senf_only = (ambiguous_events_in_senf_only_final_count / total_events_in_senf_only_final) * 100
                    logging.info(f"  Of which, events with originally ambiguous classes (list/array > 1): {ambiguous_events_in_senf_only_final_count} ({perc_ambiguous_in_senf_only:.1f}%)")
                else:
                    logging.info(f"  Of which, events with originally ambiguous classes (list/array > 1): 0 (0.0%)")

                isolated_senf_events_final_df = senf_gdf_final[senf_gdf_final['community_id'].isna()]
                
                if num_senf_only_communities_final > 0:
                    community_sizes = senf_gdf_final.groupby('community_id').size()
                    single_member_senf_communities = community_sizes[community_sizes == 1].index
                    single_member_senf_events_df = senf_gdf_final[
                        senf_gdf_final['community_id'].isin(single_member_senf_communities) & 
                        senf_gdf_final['community_id'].notna() & 
                        senf_gdf_final['community_id'].isin(final_senf_only_community_ids)
                    ]
                    if not single_member_senf_events_df.empty:
                         isolated_senf_events_final_df = pd.concat([isolated_senf_events_final_df, single_member_senf_events_df]).drop_duplicates(subset=['uid'] if 'uid' in senf_gdf_final else None)

                total_isolated_senf_final = len(isolated_senf_events_final_df)
                ambiguous_isolated_senf_final_count = 0
                for _, row in isolated_senf_events_final_df.iterrows():
                    class_val = row.get('class')
                    # MODIFIED AMBIGUITY CHECK
                    if isinstance(class_val, (list, np.ndarray)) and len(class_val) > 1:
                        ambiguous_isolated_senf_final_count += 1
                
                logging.info(f"Total isolated Senf&Seidl events (NaN community_id or alone in S&S-only community): {total_isolated_senf_final}")
                if total_isolated_senf_final > 0:
                    perc_ambiguous_isolated = (ambiguous_isolated_senf_final_count / total_isolated_senf_final) * 100
                    logging.info(f"  Of which, events with originally ambiguous classes (list/array > 1): {ambiguous_isolated_senf_final_count} ({perc_ambiguous_isolated:.1f}%)")
                else:
                     logging.info(f"  Of which, events with originally ambiguous classes (list/array > 1): 0 (0.0%)")
                logging.info(f"-----------------------------------------------------")
            else:
                logging.warning("Could not log detailed Senf&Seidl community/isolation stats post-attribution due to missing columns (community_id, dataset, or class)." )

            output_path = OUTPUT_ATTRIBUTION_DIR / "final_attributed_disturbances.parquet"
            logging.info(f"Saving final attributed GeoDataFrame to {output_path}...")
            final_attributed_gdf.to_parquet(output_path)
            logging.info(f"Successfully saved final results with {len(final_attributed_gdf)} records.")

            # --- Call QC script for final attribution results ---
            try:
                logging.info(f"Generating QC plots for final attribution results...")
                qc_attribution_script_path = PROJECT_ROOT / "src" / "qc" / "plot_attribution_results.py"
                qc_attribution_output_dir = BASE_DIR / "outputs" / "qc_attribution" # Matches default in plotting script
                qc_attribution_output_dir.mkdir(parents=True, exist_ok=True)

                subprocess.run(
                    [sys.executable, str(qc_attribution_script_path), str(output_path), "--output-dir", str(qc_attribution_output_dir)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                logging.info(f"Successfully generated attribution QC plots in {qc_attribution_output_dir}")
            except FileNotFoundError:
                logging.error(f"Attribution QC script not found at {qc_attribution_script_path}. Skipping plots.")
            except subprocess.CalledProcessError as e:
                logging.error(f"Attribution QC script failed:\nStdout:\n{e.stdout}\nStderr:\n{e.stderr}")
            except Exception as e:
                logging.error(f"An error occurred while generating attribution QC plots: {e}", exc_info=True)
            # --- End QC script call for final attribution ---

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
                # final_prob_cols = [col for col in senf_events.columns if col.startswith("prob_")] # Already defined above and used for all GDF

                # if final_prob_cols: # This check is now redundant if attributed_class_final exists
                if 'attributed_class_final' in senf_events.columns and 'attribution_confidence_final' in senf_events.columns:
                    # No aggregation step needed anymore.
                    # The prob_ columns from attr.attribute() are now the final target probabilities.
                    # logging.info(f"Using final probability columns directly from attribution output: {final_prob_cols}")

                    # Determine winning class and confidence based on these direct final probabilities
                    # senf_events["winning_class"] = senf_events[final_prob_cols].idxmax(axis=1).str.replace("prob_", "")
                    # senf_events["winning_confidence"] = senf_events[final_prob_cols].max(axis=1)
                    # These are now 'attributed_class_final' and 'attribution_confidence_final'

                    logging.info("--- Final Attribution Statistics for Senf&Seidl Events (Based on Standardized Columns) ---")
                    logging.info(f"Total Senf&Seidl events processed for attribution: {len(senf_events)}")
                    
                    logging.info("Distribution of attributed_class_final:")
                    winning_class_counts = senf_events["attributed_class_final"].value_counts()
                    for cls, count in winning_class_counts.items():
                        logging.info(f"  {cls}: {count} events ({count / len(senf_events) * 100:.2f}%)")

                    logging.info("Confidence of attributed_class_final (attribution_confidence_final):")
                    logging.info(f"  Min confidence: {senf_events['attribution_confidence_final'].min():.3f}")
                    logging.info(f"  Mean confidence: {senf_events['attribution_confidence_final'].mean():.3f}")
                    logging.info(f"  Median confidence: {senf_events['attribution_confidence_final'].median():.3f}")
                    logging.info(f"  Max confidence: {senf_events['attribution_confidence_final'].max():.3f}")
                    
                    low_confidence_threshold = 0.5 # Can be adjusted
                    num_low_confidence = (senf_events['attribution_confidence_final'] <= low_confidence_threshold).sum()
                    logging.info(f"Number of Senf&Seidl events with attribution_confidence_final <= {low_confidence_threshold}: {num_low_confidence} ({num_low_confidence / len(senf_events) * 100:.2f}%)")
                    logging.info("-----------------------------------------------------------")
                else:
                    logging.warning("Could not find 'attributed_class_final' or 'attribution_confidence_final' columns for Senf&Seidl events. Cannot compute final attribution stats.")
            else:
                logging.info("No Senf&Seidl events found in the final attributed data to generate stats for.")
            # --- End stats on final attribution ---

        else:
            logging.warning("Final attributed GeoDataFrame is None or empty. Nothing to save.")

        logging.info(f"Attribution pipeline completed. Timings: {timings}")

        # --- Save parameters on successful run ---
        try:
            with open(params_path, 'w') as f:
                json.dump(current_params.__dict__, f, indent=4)
            logging.info(f"Successfully saved current parameters to {params_path}")
        except Exception as e:
            logging.error(f"Could not save parameters to {params_path}: {e}")
        # --- End parameter saving ---

    except Exception as e:
        logging.error(f"Error during attribution steps: {e}", exc_info=True)

if __name__ == "__main__":
    main() 