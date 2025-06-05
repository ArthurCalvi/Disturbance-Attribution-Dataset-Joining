"""Process FORMS height rasters to detect clear-cuts.

Original data path used in the notebook:
'/Users/arthurcalvi/Data/Disturbances_maps/FORMS/'

Data can be downloaded from https://drive.google.com/drive/folders/1ytJ3N9Braip0LogY-4R7WMkfjzan0VzP, contact Martin Schwartz for access.
"""
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import numpy as np
import logging
import pandas as pd

# Import the new constants
from src.config.constants import RAW_TO_FINAL_TARGET_MAPPINGS, FINAL_TARGET_CLASSES

logger = logging.getLogger(__name__)

def _difference_rasters(r1: str, r2: str) -> np.ndarray:
    """Return binary difference array between two rasters."""
    with rasterio.open(r1) as src1, rasterio.open(r2) as src2:
        arr = src1.read(1) - src2.read(1)
        arr[(src2.read(1) > 500)] = 32767
        arr[(arr < 500)] = 32767
        arr[(arr > 32767 - 5000)] = 32767
        arr[arr == 32767] = 0
        arr[arr > 0] = 1
        return arr.astype("uint8"), src1.transform, src1.crs


def _extract_year(path: str) -> str:
    """Extract the first 4-digit year found in a file name."""
    stem_parts = Path(path).stem.split("_")
    for part in stem_parts:
        if part.isdigit() and len(part) == 4:
            return part
    raise ValueError(f"Year not found in filename: {path}")


def process_forms(
    rasters: List[str], 
    output_file: str,
    start_year_filter: Optional[int] = None,
    end_year_filter: Optional[int] = None
) -> gpd.GeoDataFrame:
    """Compute differences between consecutive rasters, filter by year, and create polygons."""
    polygons = []
    final_crs = 'EPSG:2154'
    source_crs = None

    if not rasters or len(rasters) < 2:
        logger.warning("Not enough rasters provided for FORMS processing (need at least 2). Returning empty GDF.")
        empty_cols = ['geometry', 'start_date', 'end_date', 'year', 'class', 'dataset']
        return gpd.GeoDataFrame(columns=empty_cols, geometry='geometry', crs=final_crs)

    rasters = sorted(rasters)
    logger.info(f"Processing FORMS with {len(rasters)} rasters: {rasters}")
    if start_year_filter and end_year_filter:
        logger.info(f"Applying year filter: {start_year_filter}-{end_year_filter}")

    for r1_path_str, r2_path_str in zip(rasters[:-1], rasters[1:]):
        try:
            year1_str = _extract_year(r1_path_str)
            year2_str = _extract_year(r2_path_str)
            
            event_year = int(year2_str)

            if start_year_filter is not None and end_year_filter is not None:
                if not (start_year_filter <= event_year <= end_year_filter):
                    continue

            diff_array, transform, current_crs = _difference_rasters(r1_path_str, r2_path_str)
            if source_crs is None:
                source_crs = current_crs

            for geom, val in shapes(diff_array, mask=diff_array > 0, transform=transform):
                polygons.append({
                    'geometry': shape(geom),
                    'start_date': datetime.strptime(f'09-{year1_str}', '%m-%Y'),
                    'end_date': datetime.strptime(f'05-{year2_str}', '%m-%Y'),
                    'year': event_year
                })
        except ValueError as ve:
            logger.warning(f"Could not extract year from filenames {Path(r1_path_str).name} or {Path(r2_path_str).name}: {ve}. Skipping pair.")
            continue
        except Exception as e:
            logger.error(f"Error processing raster pair {Path(r1_path_str).name}-{Path(r2_path_str).name}: {e}", exc_info=True)
            continue
    
    crs_for_gdf = source_crs if source_crs else final_crs

    if not polygons:
        logger.info("No FORMS polygons extracted after filtering and processing.")
        gdf = gpd.GeoDataFrame(
            columns=['geometry', 'start_date', 'end_date', 'year', 'class', 'dataset'],
            geometry='geometry',
            crs=crs_for_gdf
        )
    else:
        gdf = gpd.GeoDataFrame(polygons, geometry='geometry', crs=crs_for_gdf)

    if not gdf.empty:
        gdf = gdf.to_crs(final_crs)
        
        # Assign class using the mapping from constants
        raw_forms_class = 'clear-cut' # This is the implicit raw class for FORMS events
        forms_map = RAW_TO_FINAL_TARGET_MAPPINGS.get('forms')
        final_target_class_val = None

        if forms_map:
            # Try direct mapping first, then fallback to _default_ if specific key not found
            final_target_class_val = forms_map.get(raw_forms_class, forms_map.get('_default_'))
        
        if not final_target_class_val: # If forms_map is None, or keys are not found
            final_target_class_val = ['Unknown'] # Default to ['Unknown']
            logger.warning(
                f"FORMS mapping for raw class '{raw_forms_class}' or its default not found in "
                f"RAW_TO_FINAL_TARGET_MAPPINGS['forms']. Defaulting to {final_target_class_val}."
            )

        # Ensure 'class' column stores a list for each row.
        # final_target_class_val should be a list as per new constants.
        if not isinstance(final_target_class_val, list):
            logger.warning(f"Expected list for final_target_class_val from mapping, got {type(final_target_class_val)}. Converting to list: {[str(final_target_class_val)]}")
            final_target_class_val = [str(final_target_class_val)]
            
        gdf['class'] = [list(final_target_class_val) for _ in range(len(gdf))]
        gdf['dataset'] = 'forms'
        gdf['year'] = gdf['year'].astype(int)
    else:
        gdf = gdf.reindex(columns=['geometry', 'start_date', 'end_date', 'year', 'class', 'dataset'], fill_value=None)
        gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=final_crs)
        gdf['year'] = gdf['year'].astype('Int64')
        gdf['start_date'] = pd.to_datetime(gdf['start_date'])
        gdf['end_date'] = pd.to_datetime(gdf['end_date'])

    logger.info(f"FORMS processing complete. Generated {len(gdf)} polygons.")
    if output_file:
        logger.info(f"Saving FORMS processed data to {output_file}")
        gdf.to_parquet(output_file)
    return gdf
