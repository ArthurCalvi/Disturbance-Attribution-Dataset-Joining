"""Process Health Monitoring tabular data.

Original data path used in the notebook:
'/Users/arthurcalvi/Data/Disturbances_maps/Thierry Belouard & DSF/Veille_sanitaire/veille sanitaire DSF 2007_2023.xlsx'
"""
from pathlib import Path
import geopandas as gpd
import pandas as pd
import logging
from typing import Optional

# Import the new constants
from src.config.constants import RAW_TO_FINAL_TARGET_MAPPINGS, FINAL_TARGET_CLASSES

logger = logging.getLogger(__name__)

ALLOWED_TYPES = {
    "Dégât d'origine entomologique",
    "Dégât d'origine pathologique",
    "Dégât dû à des végétaux",
    "Dégât d'origine abiotique",
    "Dégât d'origine sylvicole",
    "Dégât du à une pollution",
}

ESSENCE_TRANSLATION = {
    'Épicéas': 'Spruces',
    'Chênes': 'Oaks',
    'Sapins': 'Firs',
    'Bouleaux': 'Birches',
    'Pins': 'Pines',
    'Douglas': 'Douglas',
    'Charme': 'Hornbeam',
    'Autres feuillus': 'Other Broadleaves',
    'Fruitiers': 'Fruit Trees',
    'Peupliers': 'Poplars',
    'Hêtre': 'Beech',
    'Mélèzes': 'Larches',
    'Frênes': 'Ash Trees',
    'Cèdres': 'Cedars',
    'Tremble': 'Aspen',
    'Saules': 'Willows',
    'Ormes': 'Elms',
    'Châtaignier': 'Chestnut',
    'Érables': 'Maples',
    'Arbustes': 'Shrubs',
    'Aulnes': 'Alders',
    'Noyers': 'Walnuts',
    'Autres résineux': 'Other Conifers',
    'Taxodiacées': 'Taxodiaceae (a family of coniferous trees)',
    'Tilleuls': 'Lindens',
}


def _get_class(row: pd.Series) -> str:
    if 'sécheresse' in row['LIB_Problème principal'].lower():
        return 'Drought-dieback'
    if 'feu ' in row['LIB_Problème principal'].lower():
        return 'Fire'
    if 'vent ' in row['LIB_Problème principal'].lower():
        return 'Storm'
    if row['LIB_Type de problème'] in {
        "Dégât d'origine entomologique",
        "Dégât d'origine pathologique",
        "Dégât dû à des végétaux",
    }:
        return 'Biotic'
    if row['LIB_Type de problème'] in {
        "Dégât d'origine abiotique",
        "Dégât d'origine sylvicole",
        "Dégât du à une pollution",
    }:
        return 'Other'
    return 'Other'


def process_hm(
    parquet_path: str, 
    output_file: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> gpd.GeoDataFrame:
    """Simplify health monitoring dataset, filter by year, and save parquet."""
    logger.info(f"Starting Health Monitoring processing from: {parquet_path}")
    if start_year and end_year:
        logger.info(f"Applying date filter: {start_year}-{end_year}")

    try:
        gdf = gpd.read_parquet(parquet_path)

        # Ensure required columns exist before attempting to use them
        required_cols_initial = ['Sévérité', 'LIB_Type de problème', "Année d'observation", 'geometry']
        missing_cols = [col for col in required_cols_initial if col not in gdf.columns]
        if missing_cols:
            logger.error(f"Missing required initial columns in HM data: {missing_cols}. Returning empty GDF.")
            # Define schema for empty GDF based on expected final output
            final_cols_schema = ['year', 'geometry', 'cause', 'notes', 'severity', 'class', 'essence', 'tree_type', 'dataset', 'start_date', 'end_date']
            return gpd.GeoDataFrame(columns=final_cols_schema, geometry='geometry', crs='EPSG:2154')

        gdf = gdf[gdf['Sévérité'] > 1].copy()
        gdf = gdf[gdf['LIB_Type de problème'].isin(ALLOWED_TYPES)].copy()

        if gdf.empty:
            logger.info("HM GeoDataFrame is empty after initial filtering (severity > 1, allowed types).")
            # Fall through to return an empty GDF with the correct schema
        else:
            gdf['year'] = gdf["Année d'observation"].astype(int)

            # Apply year filtering early if possible
            if start_year is not None and end_year is not None:
                initial_count = len(gdf)
                gdf = gdf[(gdf['year'] >= start_year) & (gdf['year'] <= end_year)]
                logger.info(f"Filtered HM data by year ({start_year}-{end_year}): {initial_count} -> {len(gdf)} records.")

            if gdf.empty:
                logger.info("HM GeoDataFrame is empty after year filtering.")
                # Fall through to return an empty GDF with the correct schema
            else:
                # Step 1: Get raw class using existing _get_class logic
                gdf['raw_class_hm'] = gdf.apply(_get_class, axis=1)
                
                # Step 2: Map this raw_class_hm to intermediate standardized class
                # We need a sub-mapping for the outputs of _get_class to our intermediate classes.
                # This is slightly different from other datasets as _get_class already does some standardization.
                # We will use the 'hm' mapping in constants.py, but the keys need to match _get_class outputs.
                # For now, let's create a specific mapping here or ensure constants.py reflects these outputs.

                # Option A: Define a local map for _get_class outputs if they are simple & few.
                # Example: {'Drought-dieback': 'drought-dieback', 'Fire': 'fire', ...}
                # Option B: Ensure RAW_TO_FINAL_TARGET_MAPPINGS['hm'] uses keys like 'Drought-dieback', 'Biotic'.
                # For now, let's assume RAW_TO_FINAL_TARGET_MAPPINGS['hm'] has been updated or we add a local map.
                # Let's assume for now that RAW_TO_FINAL_TARGET_MAPPINGS['hm'] can handle outputs of _get_class.
                # This might require updating constants.py if it currently expects codes like 'CHABLIS'.
                
                hm_class_mapping = RAW_TO_FINAL_TARGET_MAPPINGS.get('hm', {})
                gdf['class'] = gdf['raw_class_hm'].map(hm_class_mapping)
                
                unmapped_raw_hm_classes = gdf[gdf['class'].isna()]['raw_class_hm'].unique()
                if len(unmapped_raw_hm_classes) > 0:
                    logger.warning(f"Unmapped raw_class_hm values from _get_class: {unmapped_raw_hm_classes}. Defaulting to 'Unknown' final target class.")
                    logger.warning("Ensure RAW_TO_FINAL_TARGET_MAPPINGS['hm'] in constants.py correctly maps these values.")
                    gdf['class'].fillna('Unknown', inplace=True) # Default to 'Unknown'
                
                gdf['essence'] = gdf[
                    'LIB_Essence regroupée (ess. concernée)'
                ].map(ESSENCE_TRANSLATION)
                gdf['tree_type'] = gdf[
                    'LIB_Feuillus/Résineux (ess. concernée)'
                ].map({'Conifère': 'conifer', 'Feuillu': 'broadleaf'})

        # Standardize CRS if not already done (though excerpts should be in 2154)
        if gdf.crs is None or gdf.crs.to_epsg() != 2154:
             logger.info(f"Reprojecting HM data from {gdf.crs} to EPSG:2154")
             gdf = gdf.to_crs('EPSG:2154')

        # Create start_date and end_date from 'year' for schema consistency
        # Assuming the observation year represents both start and end for point data.
        if 'year' in gdf.columns and not gdf.empty:
            # For robustness, handle cases where year might be object or float before converting to int
            gdf['year'] = pd.to_numeric(gdf['year'], errors='coerce').dropna().astype(int)
            gdf['start_date'] = pd.to_datetime(gdf['year'].astype(str) + '-01-01', errors='coerce') # Default to Jan 1st
            gdf['end_date'] = pd.to_datetime(gdf['year'].astype(str) + '-12-31', errors='coerce')   # Default to Dec 31st
        elif 'year' not in gdf.columns and not gdf.empty : # Should not happen if processing went well
             logger.warning("'year' column missing before date creation for HM data.")
             # Create dummy dates if year is missing to maintain schema, though data might be compromised
             gdf['start_date'] = pd.NaT
             gdf['end_date'] = pd.NaT

        cols_to_select = [
            'year',
            'geometry',
            'LIB_Problème principal',
            'Remarques',
            'Sévérité',
            'class',
            'essence',
            'tree_type',
            'start_date', # Added for schema consistency
            'end_date'    # Added for schema consistency
        ]
        # Filter for columns that actually exist in gdf to avoid errors with partially formed gdfs
        existing_cols_to_select = [col for col in cols_to_select if col in gdf.columns]
        gdf = gdf[existing_cols_to_select]
        
        gdf.rename(
            columns={
                'LIB_Problème principal': 'cause',
                'Remarques': 'notes',
                'Sévérité': 'severity',
            },
            inplace=True,
        )
        gdf['dataset'] = 'hm'
        
        # Drop rows with NaT in essential date columns if they were created
        if 'start_date' in gdf.columns:
            gdf.dropna(subset=['start_date'], inplace=True)
        if 'end_date' in gdf.columns:
            gdf.dropna(subset=['end_date'], inplace=True) 
        # No, this was too aggressive: gdf.dropna(inplace=True) # Original dropna might remove too much if optional fields are NaN

        # Ensure all standard final columns are present, even if GDF is empty or some were dropped
        final_cols_ordered = ['year', 'start_date', 'end_date', 'geometry', 'cause', 'notes', 'severity', 'class', 'essence', 'tree_type', 'dataset']
        for col in final_cols_ordered:
            if col not in gdf.columns:
                if col == 'geometry':
                    gdf[col] = None # Or handle geometry specifically if it needs to be an empty GeoSeries
                else:
                    gdf[col] = pd.NA # Use pandas NA for missing data to support various types
        
        # Reorder and select final columns
        gdf = gdf[final_cols_ordered]

        # Add raw_class_hm if it exists, for debugging (can be removed later)
        if 'raw_class_hm' in gdf.columns and 'raw_class_hm' not in final_cols_ordered:
            final_cols_ordered.insert(final_cols_ordered.index('class'), 'raw_class_hm')
            gdf = gdf.reindex(columns=final_cols_ordered) # Reapply ordering with the new column

        # Ensure correct dtypes for an empty or populated dataframe before saving
        expected_dtypes = {
            'year': 'Int64', # Using nullable integer
            'start_date': 'datetime64[ns]',
            'end_date': 'datetime64[ns]',
            'cause': 'object',
            'notes': 'object',
            'severity': 'float64', # Or int if appropriate, depends on original data
            'class': 'object',
            'essence': 'object',
            'tree_type': 'object',
            'dataset': 'object'
        }
        for col, dtype in expected_dtypes.items():
            if col in gdf.columns:
                try:
                    if gdf[col].isnull().all() and dtype.startswith('datetime'):
                         gdf[col] = pd.to_datetime(gdf[col])
                    elif gdf[col].isnull().all() and dtype == 'Int64':
                        gdf[col] = gdf[col].astype(pd.Int64Dtype())    
                    else:
                        gdf[col] = gdf[col].astype(dtype)
                except Exception as e:
                    logger.warning(f"Could not cast column '{col}' to '{dtype}': {e}")

        logger.info(f"Health Monitoring processing complete. Generated {len(gdf)} records.")

    except Exception as e:
        logger.error(f"Error during Health Monitoring preprocessing: {e}", exc_info=True)
        # Define schema for empty GDF based on expected final output
        final_cols_schema = ['year', 'geometry', 'cause', 'notes', 'severity', 'class', 'essence', 'tree_type', 'dataset', 'start_date', 'end_date']
        gdf = gpd.GeoDataFrame(columns=final_cols_schema, geometry='geometry', crs='EPSG:2154')
        # Ensure dtypes for the empty GDF as well
        for col in final_cols_schema:
            if col == 'year': gdf[col] = gdf[col].astype(pd.Int64Dtype())
            elif col in ['start_date', 'end_date']: gdf[col] = pd.to_datetime(gdf[col])
            elif col != 'geometry': gdf[col] = gdf[col].astype(object) 

    if output_file:
        logger.info(f"Saving Health Monitoring processed data to {output_file}")
        gdf.to_parquet(output_file)
    return gdf
