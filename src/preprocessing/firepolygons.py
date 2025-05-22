"""Process fire polygons from FFUD inventory.

Original data paths used in the notebook:
 - CSV metadata: '/Users/arthurcalvi/Data/Disturbances_maps/FirePolygons/FFUD_Arthur/FFUD_Inventory_Arthur.csv'
 - Polygon directory: '/Users/arthurcalvi/Data/Disturbances_maps/FirePolygons/FFUD_Arthur/Fire_folder'
"""
from pathlib import Path
import geopandas as gpd
import pandas as pd


def process_firepolygons(csv_file: str, polygon_dir: str, output_file: str) -> gpd.GeoDataFrame:
    """Join fire polygons with attributes and save parquet."""
    df = pd.read_csv(csv_file, sep=';')
    uid_polygons = []
    for gpkg in Path(polygon_dir).glob('*.gpkg'):
        gdf = gpd.read_file(gpkg).to_crs('EPSG:2154')
        uid_polygons.append(gdf)
    if not uid_polygons:
        gdf_poly = gpd.GeoDataFrame(columns=['UID', 'geometry'], geometry='geometry', crs='EPSG:2154')
    else:
        gdf_poly = gpd.GeoDataFrame(pd.concat(uid_polygons, ignore_index=True), geometry='geometry', crs='EPSG:2154')

    df_polygons = df[df['UID'].isin(gdf_poly['UID'])]
    df_polygons = df_polygons.merge(
        gdf_poly[['UID', 'geometry']], on='UID', how='left'
    )
    gdf = gpd.GeoDataFrame(df_polygons, geometry='geometry', crs='EPSG:2154')

    gdf.rename(
        columns={
            'annee': 'year',
            'date_de_premiere_alerte': 'start_date',
            'surface_foret_m2': 'forest_area_m2',
            'nom_de_la_commune': 'name',
            'type_de_peuplement': 'essence',
        },
        inplace=True,
    )

    cols = ['year', 'code_insee', 'nom_de_la_commune', 'start_date', 'numero']
    gdf['uuid'] = gdf[cols].astype(str).agg('_'.join, axis=1)
    gdf['dataset'] = 'firepolygons'
    if output_file:
        gdf[
            [
                'uuid',
                'year',
                'start_date',
                'forest_area_m2',
                'essence',
                'name',
                'geometry',
            ]
        ].to_parquet(output_file)
    return gdf[
        ['uuid', 'year', 'start_date', 'forest_area_m2', 'essence', 'name', 'geometry']
    ]
