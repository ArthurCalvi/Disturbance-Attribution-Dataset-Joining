"""Process Combined Drought Indicator raster excerpts.

Original data path used in the notebook:
'/Users/arthurcalvi/Data/Disturbances_maps/Copernicus_CDI/CDI_2012_2023/france_extent'
"""
from pathlib import Path
from datetime import datetime
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import logging
import pandas as pd
from typing import Optional

# Import the new constants
from src.config.constants import RAW_TO_FINAL_TARGET_MAPPINGS, FINAL_TARGET_CLASSES

logger = logging.getLogger(__name__)

def process_cdi(
    input_dir: str, 
    output_file: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> gpd.GeoDataFrame:
    """Convert CDI raster files to polygons (value >=7), filter by year, and save parquet."""
    input_path = Path(input_dir)
    polygons = []
    final_crs = 'EPSG:2154'
    source_crs = None

    if start_year and end_year:
        logger.info(f"Applying date filter: {start_year}-{end_year}")

    for tif in sorted(input_path.glob('*.tif')):
        try:
            date_str = tif.stem.split('_')[-2]
            event_date = datetime.strptime(date_str, '%Y%m%d')
            event_year = event_date.year

            if start_year is not None and end_year is not None:
                if not (start_year <= event_year <= end_year):
                    continue
            
            with rasterio.open(tif) as src:
                if source_crs is None:
                    source_crs = src.crs
                image = src.read(1)
                mask = image >= 7
                if mask.any():
                    for geom, val in shapes(image, mask=mask, transform=src.transform):
                        if val >= 7:
                            polygons.append({
                                'geometry': shape(geom),
                                'start_date': event_date,
                                'end_date': event_date,
                                'year': event_year
                            })
        except ValueError as ve:
            logger.warning(f"Could not parse date from filename {tif.name}: {ve}. Skipping file.")
            continue
        except Exception as e:
            logger.error(f"Error processing file {tif.name}: {e}", exc_info=True)
            continue

    crs_for_gdf = source_crs if source_crs else 'EPSG:3035'

    if not polygons:
        logger.info("No CDI polygons extracted after filtering and processing.")
        gdf = gpd.GeoDataFrame(
            polygons,
            columns=['geometry', 'start_date', 'end_date', 'year', 'class', 'dataset'],
            geometry='geometry',
            crs=crs_for_gdf 
        )
        gdf['year'] = gdf['year'].astype('Int64')
        gdf['start_date'] = pd.to_datetime(gdf['start_date'])
        gdf['end_date'] = pd.to_datetime(gdf['end_date'])
    else:
        gdf = gpd.GeoDataFrame(polygons, geometry='geometry', crs=crs_for_gdf)
    
    if not gdf.empty:
        logger.info(f"Extracted {len(gdf)} CDI polygons before CRS conversion and finalization.")
        gdf = gdf.to_crs(final_crs)
        
        # Assign class using the mapping from constants
        raw_cdi_class = 'drought' # This is the implicit raw class for CDI events
        cdi_class_mapping = RAW_TO_FINAL_TARGET_MAPPINGS.get('cdi', {})
        final_target_class = cdi_class_mapping.get(raw_cdi_class, 'Unknown') # Default to 'Unknown'
        
        if final_target_class == 'Unknown' and raw_cdi_class in cdi_class_mapping:
            logger.warning(f"CDI raw class '{raw_cdi_class}' mapped to 'Unknown' but was in constants. Check mapping.")
        elif final_target_class == 'Unknown':
            logger.warning(f"CDI raw class '{raw_cdi_class}' not found in RAW_TO_FINAL_TARGET_MAPPINGS for 'cdi'. Defaulting to 'Unknown'.")
            
        gdf['class'] = final_target_class
        gdf['dataset'] = 'cdi'
        gdf['year'] = gdf['year'].astype(int)
    else:
        gdf = gdf.reindex(columns=['geometry', 'start_date', 'end_date', 'year', 'class', 'dataset'], fill_value=None)
        gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs=final_crs)
        gdf['year'] = gdf['year'].astype('Int64')
        gdf['start_date'] = pd.to_datetime(gdf['start_date'])
        gdf['end_date'] = pd.to_datetime(gdf['end_date'])

    logger.info(f"CDI processing complete. Generated {len(gdf)} polygons.")
    if output_file:
        logger.info(f"Saving CDI processed data to {output_file}")
        gdf.to_parquet(output_file)
    return gdf
