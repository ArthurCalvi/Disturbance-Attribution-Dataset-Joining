import os
import logging
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box
import geopandas as gpd
import numpy as np
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

# --- Constants ---
FORMS_RASTER_DIR_PATH = '/Users/arthurcalvi/Data/Disturbances_maps/FORMS/'
EXPECTED_RASTER_CRS = "EPSG:2154"
OUTPUT_DIR_RELATIVE = "../excerpts/raw/forms/"
OUTPUT_DTYPE = 'int16'
# Use a nodata value that is safe for the specified integer data type
NODATA_VAL = np.iinfo(OUTPUT_DTYPE).min

def create_raster_excerpt(input_raster_path, output_raster_path, bbox_geom, bbox_crs):
    """
    Crops a single raster file to a predefined BBOX and saves the excerpt.
    Returns True on success, False on failure, and 'skipped' if there's no overlap.
    """
    logging.debug(f"Processing: {input_raster_path}")
    try:
        with rasterio.open(input_raster_path) as src:
            raster_crs = src.crs
            if not raster_crs:
                logging.warning(f"Raster {os.path.basename(input_raster_path)} has no CRS. Assuming {EXPECTED_RASTER_CRS}.")
                raster_crs = rasterio.CRS.from_string(EXPECTED_RASTER_CRS)

            # Ensure the BBOX is in the same CRS as the raster
            if str(raster_crs) != bbox_crs:
                 bbox_gdf = gpd.GeoDataFrame([{'geometry': bbox_geom}], crs=bbox_crs)
                 bbox_geom_in_src_crs = bbox_gdf.to_crs(raster_crs).iloc[0].geometry
            else:
                bbox_geom_in_src_crs = bbox_geom

            # Perform the crop operation
            try:
                out_image, out_transform = rio_mask(src, [bbox_geom_in_src_crs], crop=True, nodata=src.nodata)
            except ValueError as e:
                if "Input shapes do not overlap raster" in str(e):
                    logging.debug(f"BBOX does not overlap with {os.path.basename(input_raster_path)}. Skipping file.")
                    return "skipped"
                else:
                    raise e # Re-raise other ValueErrors

            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "crs": raster_crs,
                "dtype": OUTPUT_DTYPE,
                "nodata": NODATA_VAL,
                "compress": "lzw"
            })

            # Create an output image with the target nodata value, then fill in valid data
            img_to_write = np.full(out_image.shape, NODATA_VAL, dtype=OUTPUT_DTYPE)
            if src.nodata is not None:
                valid_data_mask = out_image != src.nodata
                img_to_write[valid_data_mask] = out_image[valid_data_mask]
            else: # If source has no nodata, assume all data is valid
                img_to_write = out_image.astype(OUTPUT_DTYPE)

            with rasterio.open(output_raster_path, "w", **out_meta) as dest:
                dest.write(img_to_write)

            logging.debug(f"Saved excerpt: {output_raster_path}")
            return True

    except Exception as e:
        logging.error(f"Failed to process {input_raster_path}: {e}", exc_info=True)
        return False

def main(bbox_name):
    """Main function to create FORMS raster excerpts for a given bounding box."""
    logging.debug(f"Starting FORMS raster excerpt creation process for bbox: '{bbox_name}'.")

    selected_bbox = EXCERPT_BOUNDING_BOXES[bbox_name]
    bbox_geom = box(*selected_bbox['coords'])
    bbox_crs = selected_bbox['crs']

    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_output_dir = os.path.abspath(os.path.join(script_dir, OUTPUT_DIR_RELATIVE))
    os.makedirs(abs_output_dir, exist_ok=True)
    logging.debug(f"Output directory for excerpts: {abs_output_dir}")

    if not os.path.isdir(FORMS_RASTER_DIR_PATH):
        logging.error(f"Source directory not found: {FORMS_RASTER_DIR_PATH}")
        sys.exit(1)

    raster_files = [f for f in os.listdir(FORMS_RASTER_DIR_PATH) if f.lower().endswith(('.tif', '.tiff'))]
    if not raster_files:
        logging.warning(f"No TIFF files found in {FORMS_RASTER_DIR_PATH}. Exiting.")
        return
        
    processed_files, failed_files, skipped_files = 0, 0, 0
    
    bbox_name_fs = bbox_name.replace(' ', '_')

    for filename in tqdm(raster_files, desc="Processing FORMS rasters", unit="file"):
        input_path = os.path.join(FORMS_RASTER_DIR_PATH, filename)
        output_filename = f"excerpt_{bbox_name_fs}_{filename}"
        output_path = os.path.join(abs_output_dir, output_filename)

        if os.path.exists(output_path):
            logging.debug(f"Output file already exists, skipping: {output_path}")
            skipped_files += 1
            continue

        result = create_raster_excerpt(input_path, output_path, bbox_geom, bbox_crs)
        if result is True:
            processed_files += 1
        elif result == "skipped":
            skipped_files += 1
        else: # False
            failed_files += 1
            
    logging.info("--- Summary ---")
    logging.info(f"Successfully created {processed_files} excerpts.")
    logging.info(f"Skipped {skipped_files} excerpts (no overlap or already exist).")
    logging.info(f"Failed to create {failed_files} excerpts.")
    logging.debug("FORMS raster excerpt creation finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create raster excerpts for FORMS data based on a named bounding box.")
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