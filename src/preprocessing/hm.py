"""Process Health Monitoring tabular data.

Original data path used in the notebook:
'/Users/arthurcalvi/Data/Disturbances_maps/Thierry Belouard & DSF/Veille_sanitaire/veille sanitaire DSF 2007_2023.xlsx'
"""
from pathlib import Path
import geopandas as gpd
import pandas as pd


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


def process_hm(parquet_path: str, output_file: str) -> gpd.GeoDataFrame:
    """Simplify health monitoring dataset and save parquet."""
    gdf = gpd.read_parquet(parquet_path)
    gdf = gdf[gdf['Sévérité'] > 1]
    gdf = gdf[gdf['LIB_Type de problème'].isin(ALLOWED_TYPES)]

    gdf['class'] = gdf.apply(_get_class, axis=1)
    gdf['essence'] = gdf[
        'LIB_Essence regroupée (ess. concernée)'
    ].map(ESSENCE_TRANSLATION)
    gdf['tree_type'] = gdf[
        'LIB_Feuillus/Résineux (ess. concernée)'
    ].map({'Conifère': 'conifer', 'Feuillu': 'broadleaf'})

    gdf = gdf.to_crs('EPSG:2154')
    gdf['year'] = gdf["Année d'observation"].astype(int)

    cols = [
        'year',
        'geometry',
        'LIB_Problème principal',
        'Remarques',
        'Sévérité',
        'class',
        'essence',
        'tree_type',
    ]
    gdf = gdf[cols]
    gdf.rename(
        columns={
            'LIB_Problème principal': 'cause',
            'Remarques': 'notes',
            'Sévérité': 'severity',
        },
        inplace=True,
    )
    gdf['dataset'] = 'hm'
    gdf.dropna(inplace=True)

    if output_file:
        gdf.to_parquet(output_file)
    return gdf
