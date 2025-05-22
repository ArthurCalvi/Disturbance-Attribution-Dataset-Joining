import os
import logging
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom
from shapely.geometry import box
import geopandas as gpd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Path to the directory containing consolidated CDI GeoTIFF files (e.g., cdi_yyyyMMdd.tif)
CDI_RASTER_DIR_PATH = '/Users/arthurcalvi/Data/Disturbances_maps/Copernicus_CDI/CDI_2012_2023/'

# BBOX coordinates (minx, miny, maxx, maxy) in EPSG:2154
BBOX_COORDS_EPSG2154 = (307783.0822, 6340505.4366, 469246.8845, 6419190.9011)
BBOX_CRS_EPSG2154 = "EPSG:2154"
EXPECTED_RASTER_CRS_EPSG3035 = "EPSG:3035"

# Output directory relative to this script's location
OUTPUT_DIR_RELATIVE = "../excerpts/raw_raster/cdi/"
OUTPUT_DTYPE = "uint8"  # CDI values are typically small integers (0-4)
NODATA_VAL = 255         # Using a common nodata value for uint8 if original is different or not set

# File size limits in MB
MAX_SIZE_MB_WARN = 5.0
MAX_SIZE_MB_CRITICAL = 10.0

def create_cdi_raster_excerpts():
    """Crops CDI raster files to a predefined BBOX and saves them as excerpts."""
    logging.info(f"Starting CDI raster excerpt creation.")
    logging.info(f"Source CDI raster directory: {CDI_RASTER_DIR_PATH}")

    # Create the absolute path for the output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_output_dir = os.path.abspath(os.path.join(script_dir, OUTPUT_DIR_RELATIVE))
    os.makedirs(abs_output_dir, exist_ok=True)
    logging.info(f"Output directory for excerpts: {abs_output_dir}")

    # Define the BBOX geometry in EPSG:2154
    bbox_geom_epsg2154 = box(*BBOX_COORDS_EPSG2154)

    if not os.path.isdir(CDI_RASTER_DIR_PATH):
        logging.error(f"Source directory not found: {CDI_RASTER_DIR_PATH}")
        return

    processed_files = 0
    for filename in os.listdir(CDI_RASTER_DIR_PATH):
        if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
            input_raster_path = os.path.join(CDI_RASTER_DIR_PATH, filename)
            output_raster_name = f"excerpt_{filename}"
            output_raster_path = os.path.join(abs_output_dir, output_raster_name)

            logging.info(f"Processing: {input_raster_path}")

            try:
                with rasterio.open(input_raster_path) as src:
                    raster_crs = src.crs
                    if not raster_crs:
                        logging.warning(f"Raster {filename} has no CRS. Assuming {EXPECTED_RASTER_CRS_EPSG3035}.")
                        raster_crs = rasterio.CRS.from_string(EXPECTED_RASTER_CRS_EPSG3035)
                    
                    if raster_crs.is_geographic:
                         logging.warning(f"Raster {filename} CRS {raster_crs} is geographic. Ensure BBOX transformation is appropriate.")
                    
                    if str(raster_crs).upper() != EXPECTED_RASTER_CRS_EPSG3035.upper():
                        logging.warning(f"Raster {filename} CRS is {raster_crs}, expected {EXPECTED_RASTER_CRS_EPSG3035}. Proceeding with transformation, but verify BBOX.")

                    # Transform BBOX from EPSG:2154 to the raster's CRS
                    transformed_bbox_geom = transform_geom(
                        BBOX_CRS_EPSG2154, 
                        str(raster_crs), 
                        bbox_geom_epsg2154.__geo_interface__
                    )

                    # Crop the raster
                    out_image, out_transform = rio_mask(src, [transformed_bbox_geom], crop=True, nodata=src.nodata if src.nodata is not None else NODATA_VAL)
                    out_meta = src.meta.copy()

                    # Update metadata for the cropped raster
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "crs": raster_crs, # Ensure original CRS is preserved unless explicitly reprojected
                        "dtype": OUTPUT_DTYPE,
                        "nodata": NODATA_VAL,
                        "compress": "lzw" # Apply LZW compression
                    })
                    
                    # Convert data to target dtype, handling potential nodata conversion issues
                    if src.nodata is not None and src.nodata != NODATA_VAL:
                        out_image[out_image == src.nodata] = NODATA_VAL # Map old nodata to new nodata
                    
                    # Clip values if necessary for uint8 (CDI values are small, usually 0-4)
                    # This ensures values outside the typical CDI range don't cause issues with uint8
                    # For CDI, 0-4 are valid, other values might be nodata or errors.
                    # We assume values > 4 should also be mapped to NODATA_VAL for safety with uint8.
                    # If CDI can legitimately have values > 4 that are not nodata, this needs adjustment.
                    # For now, values are small, so direct conversion is usually fine, but this is a safeguard.
                    # out_image = np.clip(out_image, 0, 4) # Example: if only 0-4 are valid data
                    # If clipping, ensure nodata is handled: e.g., out_image[original_nodata_mask] = NODATA_VAL
                    
                    out_image_typed = out_image.astype(OUTPUT_DTYPE)

                    with rasterio.open(output_raster_path, "w", **out_meta) as dest:
                        dest.write(out_image_typed)
                    logging.info(f"Saved excerpt: {output_raster_path}")
                    processed_files += 1

                    # Check file size
                    file_size_mb = os.path.getsize(output_raster_path) / (1024 * 1024)
                    if file_size_mb > MAX_SIZE_MB_CRITICAL:
                        logging.critical(f"CRITICAL: File {output_raster_name} size is {file_size_mb:.2f} MB (exceeds {MAX_SIZE_MB_CRITICAL:.1f} MB limit!).")
                    elif file_size_mb > MAX_SIZE_MB_WARN:
                        logging.warning(f"WARNING: File {output_raster_name} size is {file_size_mb:.2f} MB (exceeds {MAX_SIZE_MB_WARN:.1f} MB warning limit). Consider further optimization if possible.")
                    else:
                        logging.info(f"File {output_raster_name} size is {file_size_mb:.2f} MB.")

            except Exception as e:
                logging.error(f"Failed to process {input_raster_path}: {e}", exc_info=True)
        else:
            logging.debug(f"Skipping non-TIFF file: {filename}")

    if processed_files == 0:
        logging.warning(f"No TIFF files found or processed in {CDI_RASTER_DIR_PATH}.")
    else:
        logging.info(f"Successfully processed {processed_files} CDI raster files.")

if __name__ == "__main__":
    create_cdi_raster_excerpts() 