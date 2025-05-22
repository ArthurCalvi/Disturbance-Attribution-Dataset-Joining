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


def process_cdi(input_dir: str, output_file: str) -> gpd.GeoDataFrame:
    """Convert CDI raster files to polygons (value >=7) and save parquet."""
    input_path = Path(input_dir)
    polygons = []

    for tif in sorted(input_path.glob('*.tif')):
        with rasterio.open(tif) as src:
            image = src.read(1)
            mask = image >= 7
            if mask.any():
                for geom, val in shapes(image, mask=mask, transform=src.transform):
                    if val >= 7:
                        polygons.append({
                            'geometry': shape(geom),
                            'start_date': datetime.strptime(tif.stem.split('_')[3], '%Y%m%d'),
                            'end_date': datetime.strptime(tif.stem.split('_')[3], '%Y%m%d'),
                        })

    gdf = gpd.GeoDataFrame(polygons, geometry='geometry', crs=src.crs if polygons else 'EPSG:3035')
    gdf = gdf.to_crs('EPSG:2154')
    gdf['class'] = 'drought'
    gdf['dataset'] = 'cdi'
    if output_file:
        gdf.to_parquet(output_file)
    return gdf
