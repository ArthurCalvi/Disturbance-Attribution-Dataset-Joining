import geopandas as gpd
from shapely.geometry import box
import os
import logging
import sys
import argparse
import importlib.util

# Add src to sys.path to allow for imports from src
# Assuming the script is in data_excerpt_creation, and src is a sibling directory
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(script_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from config.constants import EXCERPT_BOUNDING_BOXES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Path to the constants.py file to get dataset paths
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

def create_excerpt(gdf, bbox_geom, output_path, target_crs):
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
            logging.warning(f"GeoDataFrame for {output_path.split('/')[-1]} has no CRS set. Assuming {target_crs}.")
            gdf = gdf.set_crs(target_crs, allow_override=True) # Allow override if CRS is truly unknown but data is in target CRS
        elif gdf.crs != target_crs:
            logging.info(f"Reprojecting GeoDataFrame for {output_path.split('/')[-1]} from {gdf.crs} to {target_crs}.")
            gdf = gdf.to_crs(target_crs)

        # Clip the GeoDataFrame
        clipped_gdf = gdf[gdf.geometry.intersects(bbox_geom)]

        if clipped_gdf.empty:
            logging.info(f"No features found within the BBOX for {output_path.split('/')[-1]}. No excerpt created.")
            return False
        
        # Save the clipped GeoDataFrame
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        clipped_gdf.to_parquet(output_path)
        logging.info(f"Successfully created excerpt: {output_path}")

        return True

    except Exception as e:
        logging.error(f"Error processing dataset for {output_path.split('/')[-1]}: {e}")
        return False

def main(bbox_name):
    """
    Main function to create vector data excerpts.
    """
    logging.debug("Starting data excerpt creation process...")
    logging.debug(f"Using bounding box: '{bbox_name}'")

    # Get BBOX details from constants
    selected_bbox = EXCERPT_BOUNDING_BOXES[bbox_name]
    bbox_coords = selected_bbox['coords']
    target_crs = selected_bbox['crs']

    # Resolve paths based on the location of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    constants_abs_path = os.path.abspath(os.path.join(script_dir, CONSTANTS_FILE_PATH))
    
    # The output directory for excerpts will be relative to the directory of constants.py
    constants_dir = os.path.dirname(constants_abs_path)
    output_dir_abs = os.path.abspath(os.path.join(constants_dir, OUTPUT_DIR_RELATIVE_TO_CONSTANTS))

    logging.debug(f"Attempting to load dataset paths from: {constants_abs_path}")
    loading_dict = load_loading_dict_from_file(constants_abs_path)

    if not loading_dict:
        logging.error("Failed to load dataset paths. Exiting.")
        sys.exit(1)

    logging.debug(f"Found {len(loading_dict)} datasets to process.")
    logging.debug(f"Output directory for excerpts: {output_dir_abs}")
    os.makedirs(output_dir_abs, exist_ok=True)

    # Create a bounding box geometry
    bbox_geometry = box(*bbox_coords)
    # Create a GeoDataFrame for the bbox to ensure it has a CRS for consistent operations
    bbox_gdf = gpd.GeoDataFrame([{'geometry': bbox_geometry}], crs=target_crs)

    # Sanitize bbox_name for the filename
    bbox_name_fs = bbox_name.replace(' ', '_')

    success_count = 0
    fail_count = 0

    for dataset_name, relative_parquet_path in loading_dict.items():
        logging.debug(f"--- Processing dataset: {dataset_name} ---")
        
        # Construct the absolute path to the Parquet file
        # The paths in loading_dict are relative to the constants.py file's directory (join-datasets)
        parquet_abs_path = os.path.abspath(os.path.join(constants_dir, relative_parquet_path))
        
        output_filename = f"excerpt_{bbox_name_fs}_{os.path.basename(relative_parquet_path)}"
        output_path = os.path.join(output_dir_abs, output_filename)

        logging.debug(f"Input Parquet path: {parquet_abs_path}")
        logging.debug(f"Output excerpt path: {output_path}")

        try:
            gdf = gpd.read_parquet(parquet_abs_path)
            logging.debug(f"Successfully loaded {dataset_name} with {len(gdf)} features.")
            
            if create_excerpt(gdf, bbox_gdf.geometry.iloc[0], output_path, target_crs):
                success_count += 1
            else:
                fail_count +=1

        except FileNotFoundError:
            logging.error(f"File not found: {parquet_abs_path}. Skipping this dataset.")
            fail_count +=1
        except Exception as e:
            logging.error(f"Failed to load or process {parquet_abs_path}: {e}")
            fail_count +=1
        logging.debug(f"--- Finished processing dataset: {dataset_name} ---")


    logging.info("--- Summary ---")
    logging.info(f"Total datasets processed: {len(loading_dict)}")
    logging.info(f"Successfully created excerpts: {success_count}")
    logging.info(f"Failed/skipped excerpts: {fail_count}")
    logging.info("Data excerpt creation process finished.")

# --- Main Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create vector data excerpts based on a named bounding box.")
    parser.add_argument(
        '--bbox-name',
        type=str,
        default='les landes',
        choices=EXCERPT_BOUNDING_BOXES.keys(),
        help=f"The name of the bounding box to use for the excerpt, defined in src/config/constants.py. "
             f"Choices: {list(EXCERPT_BOUNDING_BOXES.keys())}"
    )
    args = parser.parse_args()
    main(args.bbox_name) 