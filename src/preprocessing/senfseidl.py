from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

CAUSE_MAP = {
    1: 'Storm,Biotic',
    2: 'Fire',
    3: 'Other',
}

def process_senfseidl(
    attribution_raster: str,
    year_raster: str,
    output_file: str,
) -> gpd.GeoDataFrame:
    """Join cause and year rasters and save parquet."""
    with rasterio.open(attribution_raster) as src_cause, rasterio.open(year_raster) as src_year:
        year_data = src_year.read(1)
        cause_data = src_cause.read(1)
        mask = (year_data != src_year.nodata) & (cause_data > 0)
        polygons = []
        for geom, val in shapes(year_data, mask=mask, transform=src_year.transform):
            geom = shape(geom)
            r, c = src_cause.index(*geom.centroid.coords[0])
            cause_val = int(cause_data[r, c])
            polygons.append({'geometry': geom, 'year': int(val), 'cause': cause_val})

    gdf = gpd.GeoDataFrame(polygons, geometry='geometry', crs=src_year.crs).to_crs('EPSG:2154')
    gdf['class'] = gdf['cause'].map(CAUSE_MAP)
    gdf = gdf[['year', 'geometry', 'class']]
    gdf['dataset'] = 'senfseidl'
    if output_file:
        gdf.to_parquet(output_file)
    return gdf
