from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import numpy as np
import numpy.ma as ma
import logging
import pandas as pd
from typing import Optional

# Import the new constants
from src.config.constants import RAW_TO_FINAL_TARGET_MAPPINGS, FINAL_TARGET_CLASSES

# Configure logging for this module
logger = logging.getLogger(__name__)

# This map translates the numeric cause codes from the Senf & Seidl raster
# to the descriptive raw cause strings that are keys in RAW_TO_INTERMEDIATE_MAPPINGS['senfseidl']
# This needs to align with senf_seidl_legend.csv or similar documentation.
# UPDATED TO SIMPLER MAPPING based on user feedback and notebook snippet
SENFSEIDL_CODE_TO_RAW_CAUSE = {
    1: 'Storm,Biotic', 
    2: 'Fire',
    3: 'Other',
    # Add other numeric codes if they actually appear in the cause raster excerpts
}

def process_senfseidl(
    attribution_raster_path: str,
    year_raster_path: str,
    output_file: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """Process Senf & Seidl year and cause rasters, join them, 
       dissolve by year and cause, filter by year range, and save as parquet."""
    logger.info(f"Starting Senf & Seidl processing. Year: {year_raster_path}, Cause: {attribution_raster_path}")
    if start_year and end_year:
        logger.info(f"Applying date filter: {start_year}-{end_year}")

    try:
        # --- Process Year Raster ---
        with rasterio.open(year_raster_path) as src_year:
            year_data = src_year.read(1)
            year_profile = src_year.profile
            year_crs = src_year.crs
            year_nodata = src_year.nodata if src_year.nodata is not None else np.iinfo(year_data.dtype).max
            
            logger.info(f"Year raster {Path(year_raster_path).name}: CRS={year_crs}, NoData={year_nodata}, Values={np.unique(year_data)}")
            year_mask = (year_data == year_nodata)
            year_data_ma = ma.masked_array(year_data, mask=year_mask)

            year_shapes = ((shape(s), int(v)) for s, v in shapes(year_data_ma, transform=src_year.transform))
            gdf_year = gpd.GeoDataFrame(year_shapes, columns=["geometry", "year"], crs=year_crs)
            logger.info(f"Vectorized year raster: {len(gdf_year)} initial polygons.")

            # Apply year filtering to gdf_year BEFORE any joins or further processing
            if start_year is not None and end_year is not None and 'year' in gdf_year.columns:
                initial_year_count = len(gdf_year)
                gdf_year = gdf_year[(gdf_year['year'] >= start_year) & (gdf_year['year'] <= end_year)]
                logger.info(f"Filtered gdf_year by year ({start_year}-{end_year}): {initial_year_count} -> {len(gdf_year)} polygons.")
                if gdf_year.empty:
                    logger.warning("gdf_year is empty after year filtering. No Senf & Seidl data to process for this period.")
                    # Create an empty GDF with the expected schema
                    final_gdf = gpd.GeoDataFrame(columns=['year', 'geometry', 'class', 'dataset', 'cause_value'], crs='EPSG:2154')
                    if output_file:
                        final_gdf.to_parquet(output_file)
                    return final_gdf # Exit early
            elif (start_year is not None or end_year is not None) and 'year' not in gdf_year.columns:
                logger.warning("Year filtering requested for gdf_year, but 'year' column not found.")

        # --- Process Cause Raster ---
        with rasterio.open(attribution_raster_path) as src_cause:
            cause_data = src_cause.read(1)
            cause_profile = src_cause.profile
            cause_crs = src_cause.crs
            if cause_crs is None:
                logger.warning(f"Cause raster {Path(attribution_raster_path).name} missing CRS, assuming EPSG:3035.")
                cause_crs = rasterio.crs.CRS.from_epsg(3035)
            
            cause_nodata = src_cause.nodata if src_cause.nodata is not None else np.finfo(cause_data.dtype).min
            logger.info(f"Cause raster {Path(attribution_raster_path).name}: CRS={cause_crs}, NoData={cause_nodata}, Values={np.unique(cause_data)}")
            
            cause_mask = (cause_data == cause_nodata) | (cause_data < 0) 
            cause_data_ma = ma.masked_array(cause_data, mask=cause_mask)
            
            cause_shapes = ((shape(s), int(v)) for s, v in shapes(cause_data_ma, transform=src_cause.transform))
            gdf_cause = gpd.GeoDataFrame(cause_shapes, columns=["geometry", "cause_value"], crs=cause_crs)
            logger.info(f"Vectorized cause raster: {len(gdf_cause)} initial polygons.")

        target_crs_sjoin = rasterio.crs.CRS.from_epsg(3035)
        if gdf_year.crs != target_crs_sjoin:
            logger.info(f"Reprojecting year GDF from {gdf_year.crs} to {target_crs_sjoin}")
            gdf_year = gdf_year.to_crs(target_crs_sjoin)
        if gdf_cause.crs != target_crs_sjoin:
            logger.info(f"Reprojecting cause GDF from {gdf_cause.crs} to {target_crs_sjoin}")
            gdf_cause = gdf_cause.to_crs(target_crs_sjoin)
        
        logger.info(f"Performing spatial join between year ({len(gdf_year)} polygons) and cause ({len(gdf_cause)} polygons)...")
        gdf_joined = gpd.sjoin(gdf_year, gdf_cause, how='inner', predicate='intersects')
        if 'index_right' in gdf_joined.columns:
            gdf_joined = gdf_joined.drop(columns=['index_right'])
        logger.info(f"Spatial join resulted in {len(gdf_joined)} polygons.")

        if gdf_joined.empty:
            logger.warning("Spatial join resulted in an empty GeoDataFrame. No further processing for Senf & Seidl.")
            final_gdf = gpd.GeoDataFrame(columns=['year', 'geometry', 'class', 'dataset'], crs='EPSG:2154')
            if output_file:
                final_gdf.to_parquet(output_file)
            return final_gdf

        gdf_joined['year'] = gdf_joined['year'].astype(int)
        gdf_joined['cause_value'] = gdf_joined['cause_value'].astype(int)
        
        gdf_dissolved = gdf_joined.dissolve(by=['year', 'cause_value'], as_index=False)
        logger.info(f"Dissolved into {len(gdf_dissolved)} polygons.")

        logger.info(f"Exploding MultiPolygons for {len(gdf_dissolved)} features...")
        gdf_exploded = gdf_dissolved.explode(index_parts=True)
        if isinstance(gdf_exploded.index, pd.MultiIndex):
             gdf_exploded = gdf_exploded.reset_index(drop=True)
        else:
            gdf_exploded = gdf_exploded.reset_index(drop=True)

        logger.info(f"Exploded into {len(gdf_exploded)} final polygons.")
        
        final_gdf = gdf_exploded

        # --- Map Senf&Seidl cause codes to final target classes ---
        # Step 1: Map numeric cause_value to raw descriptive cause string
        final_gdf['raw_cause_description'] = final_gdf['cause_value'].map(SENFSEIDL_CODE_TO_RAW_CAUSE)
        
        # Handle cases where a cause_value might not be in SENFSEIDL_CODE_TO_RAW_CAUSE
        unmapped_codes = final_gdf[final_gdf['raw_cause_description'].isna()]['cause_value'].unique()
        if len(unmapped_codes) > 0:
            logger.warning(f"Unmapped Senf&Seidl cause_values found: {unmapped_codes}. These will be mapped to 'Other' raw cause, then to its corresponding final target class.")
            final_gdf['raw_cause_description'].fillna('Other', inplace=True) # Default unmapped codes to 'Other' raw cause

        # Step 2: Map raw descriptive cause string to final target class
        senfseidl_class_mapping = RAW_TO_FINAL_TARGET_MAPPINGS.get('senfseidl', {})
        final_gdf['class'] = final_gdf['raw_cause_description'].map(senfseidl_class_mapping)

        # Handle cases where a raw_cause_description might not be in senfseidl_class_mapping
        unmapped_raw_causes = final_gdf[final_gdf['class'].isna()]['raw_cause_description'].unique()
        if len(unmapped_raw_causes) > 0:
            logger.warning(f"Unmapped Senf&Seidl raw_cause_descriptions: {unmapped_raw_causes}. Mapping to 'Unknown' final target class.")
            final_gdf['class'].fillna('Unknown', inplace=True) # Default to 'Unknown' from FINAL_TARGET_CLASSES
        
        # Ensure all classes are within the defined FINAL_TARGET_CLASSES (optional check)
        # unknown_final_target = final_gdf[~final_gdf['class'].isin(FINAL_TARGET_CLASSES)]['class'].unique()
        # if len(unknown_final_target) > 0:
        #     logger.warning(f"Senf&Seidl produced classes not in FINAL_TARGET_CLASSES: {unknown_final_target}. This is unexpected.")

        final_gdf['dataset'] = 'senfseidl'
        
        final_gdf = final_gdf[['year', 'geometry', 'class', 'dataset', 'cause_value', 'raw_cause_description']]
        # Ensure 'raw_cause_description' is kept if useful for debugging, otherwise can be dropped.

        # Apply year filtering --- THIS IS NOW REDUNDANT if applied to gdf_year earlier, but kept for safety or if logic changes
        # However, if gdf_year was filtered, this final_gdf should already be filtered.
        # Consider removing or commenting out this redundant filter if performance is critical and logic is stable.
        if start_year is not None and end_year is not None and 'year' in final_gdf.columns:
            # This log might be confusing if filtering happened earlier. 
            # Let's assume the primary filtering is on gdf_year.
            # If final_gdf is somehow different, this would catch it.
            if not gdf_year.empty: # Only log if there was something to filter initially from gdf_year stage
                 logger.debug(f"Re-checking year filter on final_gdf ({start_year}-{end_year}). Count: {len(final_gdf)} polygons.")
        elif (start_year is not None or end_year is not None) and 'year' not in final_gdf.columns:
            logger.warning("Year filtering requested, but 'year' column not found in GeoDataFrame.")

        logger.info(f"Final columns: {final_gdf.columns.tolist()}")

        final_gdf = final_gdf.to_crs('EPSG:2154')
        logger.info(f"Senf & Seidl processing complete. Generated {len(final_gdf)} polygons.")

    except Exception as e:
        logger.error(f"Error during Senf & Seidl preprocessing: {e}", exc_info=True)
        final_gdf = gpd.GeoDataFrame(columns=['year', 'geometry', 'class', 'dataset', 'cause_value'], crs='EPSG:2154')

    if output_file:
        logger.info(f"Saving Senf & Seidl processed data to {output_file}")
        final_gdf.to_parquet(output_file)
    return final_gdf
