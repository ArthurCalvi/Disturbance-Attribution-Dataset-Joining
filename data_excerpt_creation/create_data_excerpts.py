import geopandas as gpd
from shapely.geometry import box
import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# BBOX coordinates (minx, miny, maxx, maxy) in EPSG:2154
BBOX_COORDS = (307783.0822, 6340505.4366, 469246.8845, 6419190.9011)
TARGET_CRS = "EPSG:2154"

# Path to the constants.py file to get dataset paths
# Assuming this script is in data_excerpt_creation and constants.py is in join-datasets
# Adjust the path if your directory structure is different
CONSTANTS_FILE_PATH = "../join-datasets/constants.py" 
OUTPUT_DIR_RELATIVE_TO_CONSTANTS = "../excerpts/" # Relative to the constants.py file's directory

# --- Helper Functions ---
def load_loading_dict_from_file(file_path):
    """
    Loads the loading_dict from the specified Python file.
    This is a simplified way to get the dictionary.
    A more robust way would be to import it if the structure allows,
    or use ast.literal_eval if the file content is complex.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find the loading_dict assignment
        # This is a fragile way and might break if the constants.py format changes
        # For a robust solution, consider restructuring constants.py or using ast.literal_eval
        dict_str_start = content.find("loading_dict = {")
        if dict_str_start == -1:
            logging.error(f"Could not find 'loading_dict = {{' in {file_path}")
            return None
        
        # Try to find the matching closing brace
        open_braces = 0
        dict_str_end = -1
        for i, char in enumerate(content[dict_str_start:]):
            if char == '{':
                open_braces += 1
            elif char == '}':
                open_braces -= 1
                if open_braces == 0:
                    dict_str_end = dict_str_start + i + 1
                    break
        
        if dict_str_end == -1:
            logging.error(f"Could not find the closing brace for loading_dict in {file_path}")
            return None

        loading_dict_str = content[dict_str_start + len("loading_dict = "):dict_str_end]
        
        # Safely evaluate the dictionary string
        # We need to ensure 'sys.path' allows importing custom modules if constants.py relies on them
        # For simplicity here, we assume it's a direct dictionary definition
        # A better way if constants.py could be imported:
        # import importlib.util
        # spec = importlib.util.spec_from_file_location("constants_module", file_path)
        # constants_module = importlib.util.module_from_spec(spec)
        # spec.loader.exec_module(constants_module)
        # return constants_module.loading_dict
        
        # Using exec, be cautious if the file content is not trusted
        local_vars = {}
        exec(f"loading_dict = {loading_dict_str}", {}, local_vars)
        return local_vars.get('loading_dict')

    except FileNotFoundError:
        logging.error(f"Constants file not found at {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading loading_dict from {file_path}: {e}")
        return None

def create_excerpt(gdf, bbox_geom, output_path):
    """
    Clips a GeoDataFrame to the bbox_geom and saves it.
    """
    try:
        # Ensure the GeoDataFrame has a geometry column
        if 'geometry' not in gdf.columns or gdf.geometry.isnull().all():
            logging.warning(f"No valid geometry column found or all geometries are null in {output_path.split('/')[-1]}. Skipping clipping.")
            return False

        # Ensure CRS is set and matches the target CRS
        if gdf.crs is None:
            logging.warning(f"GeoDataFrame for {output_path.split('/')[-1]} has no CRS set. Assuming {TARGET_CRS}.")
            gdf = gdf.set_crs(TARGET_CRS, allow_override=True) # Allow override if CRS is truly unknown but data is in target CRS
        elif gdf.crs != TARGET_CRS:
            logging.info(f"Reprojecting GeoDataFrame for {output_path.split('/')[-1]} from {gdf.crs} to {TARGET_CRS}.")
            gdf = gdf.to_crs(TARGET_CRS)

        # Clip the GeoDataFrame
        # For points, 'within' is more appropriate. For polygons/lines, 'intersects' is often used.
        # Using 'intersects' generally works well for creating excerpts.
        # A robust way is to check geometry types, but intersects is a good default.
        clipped_gdf = gdf[gdf.geometry.intersects(bbox_geom)]

        if clipped_gdf.empty:
            logging.info(f"No features found within the BBOX for {output_path.split('/')[-1]}. No excerpt created.")
            return False
        
        # Save the clipped GeoDataFrame
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        clipped_gdf.to_parquet(output_path)
        logging.info(f"Successfully created excerpt: {output_path}")

        # Check file size and resample if necessary
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logging.info(f"Initial excerpt size for {output_path.split('/')[-1]}: {file_size_mb:.2f} MB")

        if file_size_mb > 5:
            logging.warning(
                f"Excerpt {output_path} is {file_size_mb:.2f} MB (larger than 5MB). "
                f"Attempting to resample to be < 10MB."
            )
            
            TARGET_MAX_SAMPLING_MB = 9.5  # Target size in MB for resampling

            # Only attempt to sample if the GeoDataFrame is not empty and sampling makes sense
            if not clipped_gdf.empty:
                # Calculate sampling fraction to target TARGET_MAX_SAMPLING_MB
                # This fraction applies to the number of rows, assuming size is proportional to rows.
                if file_size_mb > 0: # Avoid division by zero
                    sampling_fraction_for_size = TARGET_MAX_SAMPLING_MB / file_size_mb
                    
                    if sampling_fraction_for_size < 1.0:
                        # Apply this fraction to the number of rows
                        n_target_rows = max(1, int(len(clipped_gdf) * sampling_fraction_for_size))
                        
                        if n_target_rows < len(clipped_gdf):
                            logging.info(
                                f"Resampling {output_path.split('/')[-1]} from {len(clipped_gdf)} rows to {n_target_rows} rows "
                                f"(sampling fraction for size: {sampling_fraction_for_size:.4f})."
                            )
                            sampled_gdf = clipped_gdf.sample(n=n_target_rows, random_state=1) # random_state for reproducibility
                            sampled_gdf.to_parquet(output_path)  # Overwrite the file
                            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)  # Update file_size_mb
                            logging.info(
                                f"New excerpt size for {output_path.split('/')[-1]} after sampling: {file_size_mb:.2f} MB"
                            )
                        else:
                            logging.info(
                                f"Calculated target rows ({n_target_rows}) is not less than current rows ({len(clipped_gdf)}). "
                                f"Skipping resampling for {output_path.split('/')[-1]}. Current size: {file_size_mb:.2f} MB."
                            )
                    else:
                        logging.info(
                            f"Calculated sampling fraction for size ({sampling_fraction_for_size:.4f}) is >= 1.0. "
                            f"File is likely already < {TARGET_MAX_SAMPLING_MB}MB or sampling won't reduce size significantly. "
                            f"Skipping resampling for {output_path.split('/')[-1]}. Current size: {file_size_mb:.2f} MB."
                        )
                else:
                    logging.warning(f"File size of {output_path.split('/')[-1]} is 0MB. Cannot calculate sampling fraction.")
            else:
                # This case should ideally not be reached if the function returned False earlier for empty clipped_gdf
                logging.warning(f"Clipped GeoDataFrame for {output_path.split('/')[-1]} is empty. Cannot sample.")

        # Final size check and warning
        if file_size_mb > 10:
            logging.warning(
                f"FINAL Excerpt {output_path} is {file_size_mb:.2f} MB, which is STILL LARGER than 10MB "
                "despite any sampling attempt."
            )
        elif file_size_mb > 5: # If it's between 5 and 10 MB (inclusive of 5, exclusive of 10 from above)
             logging.warning(
                f"FINAL Excerpt {output_path} is {file_size_mb:.2f} MB (between 5MB and 10MB)."
            )
        # If <= 5MB, no warning needed here as it meets the original implicit goal.

        return True

    except Exception as e:
        logging.error(f"Error processing dataset for {output_path.split('/')[-1]}: {e}")
        return False

# --- Main Script ---
if __name__ == "__main__":
    logging.info("Starting data excerpt creation process...")

    # Resolve paths based on the location of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    constants_abs_path = os.path.abspath(os.path.join(script_dir, CONSTANTS_FILE_PATH))
    
    # The output directory for excerpts will be relative to the directory of constants.py
    constants_dir = os.path.dirname(constants_abs_path)
    output_dir_abs = os.path.abspath(os.path.join(constants_dir, OUTPUT_DIR_RELATIVE_TO_CONSTANTS))

    logging.info(f"Attempting to load dataset paths from: {constants_abs_path}")
    loading_dict = load_loading_dict_from_file(constants_abs_path)

    if not loading_dict:
        logging.error("Failed to load dataset paths. Exiting.")
        sys.exit(1)

    logging.info(f"Found {len(loading_dict)} datasets to process.")
    logging.info(f"Output directory for excerpts: {output_dir_abs}")
    os.makedirs(output_dir_abs, exist_ok=True)

    # Create a bounding box geometry
    bbox_geometry = box(*BBOX_COORDS)
    # Create a GeoDataFrame for the bbox to ensure it has a CRS for consistent operations
    bbox_gdf = gpd.GeoDataFrame([{'geometry': bbox_geometry}], crs=TARGET_CRS)


    success_count = 0
    fail_count = 0

    for dataset_name, relative_parquet_path in loading_dict.items():
        logging.info(f"--- Processing dataset: {dataset_name} ---")
        
        # Construct the absolute path to the Parquet file
        # The paths in loading_dict are relative to the constants.py file's directory (join-datasets)
        parquet_abs_path = os.path.abspath(os.path.join(constants_dir, relative_parquet_path))
        
        output_filename = f"excerpt_{os.path.basename(relative_parquet_path)}"
        output_path = os.path.join(output_dir_abs, output_filename)

        logging.info(f"Input Parquet path: {parquet_abs_path}")
        logging.info(f"Output excerpt path: {output_path}")

        try:
            gdf = gpd.read_parquet(parquet_abs_path)
            logging.info(f"Successfully loaded {dataset_name} with {len(gdf)} features.")
            
            if create_excerpt(gdf, bbox_gdf.geometry.iloc[0], output_path):
                success_count += 1
            else:
                fail_count +=1

        except FileNotFoundError:
            logging.error(f"File not found: {parquet_abs_path}. Skipping this dataset.")
            fail_count +=1
        except Exception as e:
            logging.error(f"Failed to load or process {parquet_abs_path}: {e}")
            fail_count +=1
        logging.info(f"--- Finished processing dataset: {dataset_name} ---")


    logging.info("--- Summary ---")
    logging.info(f"Total datasets processed: {len(loading_dict)}")
    logging.info(f"Successfully created excerpts: {success_count}")
    logging.info(f"Failed/skipped excerpts: {fail_count}")
    logging.info("Data excerpt creation process finished.") 