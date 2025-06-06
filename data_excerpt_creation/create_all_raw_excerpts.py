import argparse
import logging
import sys
import os
import time

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

# Now, we can import the scripts from the same directory
try:
    from . import create_cdi_raster_excerpts
    from . import create_firepolygons_excerpts
    from . import create_forms_raster_excerpt
    from . import create_healthmonitoring_vector_excerpt
    from . import create_senfseidl_raster_excerpts
except ImportError:
    # This is for standalone execution
    import create_cdi_raster_excerpts
    import create_firepolygons_excerpts
    import create_forms_raster_excerpt
    import create_healthmonitoring_vector_excerpt
    import create_senfseidl_raster_excerpts


def main(bbox_name, verbose=False):
    """
    Runs all the data excerpt creation scripts for a given bounding box.
    This script is for creating excerpts from RAW data sources.
    """
    if verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        stream=sys.stdout)

    # For robustness in file paths, replace spaces with underscores
    bbox_name_fs = bbox_name.replace(' ', '_')
    logging.info(f"--- Starting raw data excerpt creation for BBOX: '{bbox_name}' ---")

    scripts_to_run = {
        "CDI Rasters": create_cdi_raster_excerpts.main,
        "Fire Polygons": create_firepolygons_excerpts.main,
        "FORMS Rasters": create_forms_raster_excerpt.main,
        "Health Monitoring Vectors": create_healthmonitoring_vector_excerpt.main,
        "Senf-Seidl Rasters": create_senfseidl_raster_excerpts.main
    }

    for name, script_main in scripts_to_run.items():
        logging.info(f"--- Running: {name} ---")
        start_time = time.time()
        try:
            script_main(bbox_name) # Pass the original bbox_name
        except Exception as e:
            logging.error(f"An error occurred in '{name}': {e}", exc_info=True)
        end_time = time.time()
        logging.info(f"--- Finished: {name} in {end_time - start_time:.2f} seconds ---")

    logging.info(f"--- All raw data excerpt creation finished for BBOX: '{bbox_name}' ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all raw data excerpt creation scripts for a specific bounding box.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--bbox-name',
                        default='les landes',
                        help='The name of the bounding box to use for the excerpts (e.g., "les landes").\n'
                             'Default is "les landes".')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging output.')

    args = parser.parse_args()
    main(args.bbox_name, args.verbose) 