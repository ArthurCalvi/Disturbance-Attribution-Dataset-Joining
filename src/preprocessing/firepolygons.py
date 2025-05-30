"""Process fire polygons from FFUD inventory.

Original data paths used in the notebook:
 - CSV metadata: '/Users/arthurcalvi/Data/Disturbances_maps/FirePolygons/FFUD_Arthur/FFUD_Inventory_Arthur.csv'
 - Polygon directory: '/Users/arthurcalvi/Data/Disturbances_maps/FirePolygons/FFUD_Arthur/Fire_folder'
"""
from pathlib import Path
import geopandas as gpd
import pandas as pd
import logging
from typing import Optional

# Import the new constants
from src.config.constants import RAW_TO_FINAL_TARGET_MAPPINGS, FINAL_TARGET_CLASSES

logger = logging.getLogger(__name__)

def process_firepolygons(
    csv_file: str, 
    polygon_dir: str, 
    output_file: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> gpd.GeoDataFrame:
    """Join fire polygons with attributes, filter by year, and save parquet."""
    logger.info(f"Starting Fire Polygons processing. CSV: {csv_file}, Dir: {polygon_dir}")
    if start_year and end_year:
        logger.info(f"Applying date filter: {start_year}-{end_year}")

    try:
        df = pd.read_csv(csv_file, sep=';')
        uid_polygons = []
        for gpkg in Path(polygon_dir).glob('*.gpkg'):
            try:
                gdf_gpkg = gpd.read_file(gpkg).to_crs('EPSG:2154')
                if not gdf_gpkg.empty:
                    uid_polygons.append(gdf_gpkg)
                else:
                    logger.warning(f"GPKG file {gpkg.name} is empty or has no geometries after CRS conversion.")
            except Exception as e:
                logger.error(f"Error reading or processing GPKG file {gpkg.name}: {e}", exc_info=True)
                continue
        
        if not uid_polygons:
            logger.warning("No valid polygons found in the GPKG directory. Returning empty GeoDataFrame.")
            final_columns = [
                'uuid', 'year', 'start_date', 'end_date', 'class', 'dataset', 
                'forest_area_m2', 'essence', 'name', 'geometry'
            ]
            empty_schema_cols = ['uuid', 'year', 'start_date', 'end_date', 'class', 'dataset', 'geometry']
            gdf_final = gpd.GeoDataFrame(columns=empty_schema_cols, geometry='geometry', crs='EPSG:2154')
            if output_file:
                gdf_final.to_parquet(output_file)
            return gdf_final

        gdf_poly_all = gpd.GeoDataFrame(pd.concat(uid_polygons, ignore_index=True), geometry='geometry', crs='EPSG:2154')
        if 'UID' not in gdf_poly_all.columns:
            logger.error("Concatenated GPKG files do not contain 'UID' column. Cannot merge with attributes.")
            return gpd.GeoDataFrame(columns=['uuid', 'year', 'start_date', 'end_date', 'class', 'dataset', 'geometry'], geometry='geometry', crs='EPSG:2154')

        if 'UID' in df.columns:
            df['UID'] = df['UID'].astype(gdf_poly_all['UID'].dtype)
        else:
            logger.error("CSV file does not contain 'UID' column. Cannot merge with geometries.")
            return gpd.GeoDataFrame(columns=['uuid', 'year', 'start_date', 'end_date', 'class', 'dataset', 'geometry'], geometry='geometry', crs='EPSG:2154')

        df_attributes_filtered = df[df['UID'].isin(gdf_poly_all['UID'])].copy()

        gdf_merged = gdf_poly_all.merge(
            df_attributes_filtered, on='UID', how='inner'
        )

        if gdf_merged.empty:
            logger.warning("No matching records found after merging polygons and attributes.")
            return gpd.GeoDataFrame(columns=['uuid', 'year', 'start_date', 'end_date', 'class', 'dataset', 'geometry'], geometry='geometry', crs='EPSG:2154')

        gdf_merged.rename(
            columns={
                'annee': 'year',
                'date_de_premiere_alerte': 'start_date_str',
                'surface_foret_m2': 'forest_area_m2',
                'nom_de_la_commune': 'name',
                'type_de_peuplement': 'essence',
            },
            inplace=True,
        )
        
        # Convert start_date_str to datetime, coercing errors
        # Ensure 'start_date_str' column exists from the rename operation
        if 'start_date_str' in gdf_merged.columns:
            gdf_merged['start_date'] = pd.to_datetime(gdf_merged['start_date_str'], errors='coerce')
            # For fire polygons, end_date is often not explicitly available or is the same as start_date.
            # We will set end_date to be the same as start_date for consistency.
            gdf_merged['end_date'] = gdf_merged['start_date']
            # Drop rows where start_date could not be parsed (NaT)
            gdf_merged.dropna(subset=['start_date'], inplace=True)
        else:
            logger.warning("Column 'start_date_str' (expected from renaming 'date_de_premiere_alerte') not found. Cannot create date columns.")
            # Create empty/NaT date columns to maintain schema if 'start_date_str' was missing
            gdf_merged['start_date'] = pd.NaT
            gdf_merged['end_date'] = pd.NaT

        if gdf_merged.empty or 'start_date' not in gdf_merged.columns or gdf_merged['start_date'].isnull().all():
            logger.warning("No valid records after date conversion or 'start_date_str' was missing. Returning empty GeoDataFrame.")
            return gpd.GeoDataFrame(columns=['uuid', 'year', 'start_date', 'end_date', 'class', 'dataset', 'geometry'], geometry='geometry', crs='EPSG:2154')

        if 'year' in gdf_merged.columns:
            gdf_merged['year'] = gdf_merged['year'].astype(int)
        else:
            logger.warning("'year' column (from 'annee') not found after rename. Will derive from 'start_date'.")
            gdf_merged['year'] = gdf_merged['start_date'].dt.year

        if start_year is not None and end_year is not None:
            initial_count = len(gdf_merged)
            gdf_merged = gdf_merged[(gdf_merged['year'] >= start_year) & (gdf_merged['year'] <= end_year)]
            logger.info(f"Filtered by year ({start_year}-{end_year}): {initial_count} -> {len(gdf_merged)} records.")

        if gdf_merged.empty:
            logger.info("GeoDataFrame is empty after year filtering.")
            return gpd.GeoDataFrame(columns=['uuid', 'year', 'start_date', 'end_date', 'class', 'dataset', 'geometry'], geometry='geometry', crs='EPSG:2154')

        # Ensure required columns for UUID are present and are strings
        # Use 'start_date' (datetime) for UUID creation if 'start_date_str' led to NaT or was missing for some records
        # For consistency, we should decide if UUID uses original string or the parsed date string. Let's use original if available.
        uuid_cols_options = ['year', 'code_insee', 'name', 'start_date_str', 'numero'] 
        uuid_cols_to_use = []

        for col_opt in uuid_cols_options:
            if col_opt in gdf_merged.columns:
                gdf_merged[col_opt] = gdf_merged[col_opt].astype(str) # Ensure string type
                uuid_cols_to_use.append(col_opt)
            elif col_opt == 'start_date_str' and 'start_date' in gdf_merged.columns: # Fallback for date string if original is missing
                logger.warning(f"Column 'start_date_str' for UUID is missing, using formatted 'start_date'.")
                gdf_merged['__uuid_temp_date__'] = gdf_merged['start_date'].dt.strftime('%Y-%m-%d') # Or appropriate format
                uuid_cols_to_use.append('__uuid_temp_date__')
            else:
                logger.warning(f"Column '{col_opt}' needed for UUID creation is missing and no fallback. Filling with 'unknown'.")
                gdf_merged[col_opt] = 'unknown' # Add missing column with placeholder
                gdf_merged[col_opt] = gdf_merged[col_opt].astype(str)
                uuid_cols_to_use.append(col_opt)

        gdf_merged['uuid'] = gdf_merged[uuid_cols_to_use].agg('_'.join, axis=1)
        if '__uuid_temp_date__' in gdf_merged.columns:
            gdf_merged = gdf_merged.drop(columns=['__uuid_temp_date__'])
            
        gdf_merged['dataset'] = 'firepolygons'
        
        # Assign class using the mapping from constants
        # For firepolygons, we use a default raw class key since the dataset is implicitly fire.
        raw_fire_class_key = '_default_' 
        firepolygons_class_mapping = RAW_TO_FINAL_TARGET_MAPPINGS.get('firepolygons', {})
        final_target_class = firepolygons_class_mapping.get(raw_fire_class_key, 'Unknown')

        if final_target_class == 'Unknown' and raw_fire_class_key in firepolygons_class_mapping:
            logger.warning(f"Firepolygons raw class key '{raw_fire_class_key}' mapped to 'Unknown' but was in constants. Check mapping.")
        elif final_target_class == 'Unknown':
            logger.warning(f"Firepolygons raw class key '{raw_fire_class_key}' not found in RAW_TO_FINAL_TARGET_MAPPINGS for 'firepolygons'. Defaulting to 'Unknown'.")

        gdf_merged['class'] = final_target_class

        final_columns_ordered = [
            'uuid', 'year', 'start_date', 'end_date', 'class', 'dataset',
            'forest_area_m2', 'essence', 'name', 'geometry'
        ]
        gdf_final = gdf_merged[[col for col in final_columns_ordered if col in gdf_merged.columns]]

        logger.info(f"Fire Polygons processing complete. Generated {len(gdf_final)} records.")

    except Exception as e:
        logger.error(f"Error during Fire Polygons preprocessing: {e}", exc_info=True)
        gdf_final = gpd.GeoDataFrame(columns=['uuid', 'year', 'start_date', 'end_date', 'class', 'dataset', 'geometry'], geometry='geometry', crs='EPSG:2154')

    if output_file:
        logger.info(f"Saving Fire Polygons processed data to {output_file}")
        gdf_final.to_parquet(output_file)
    
    return gdf_final
