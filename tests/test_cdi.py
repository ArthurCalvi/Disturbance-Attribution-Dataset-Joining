import geopandas as gpd
from src.preprocessing.cdi import process_cdi


def test_process_cdi(tmp_path):
    gdf = process_cdi('excerpts/raw/cdi', tmp_path/'out.parquet')
    assert gdf.crs.to_string() == 'EPSG:2154'
    for col in ['geometry', 'start_date', 'end_date', 'class', 'dataset']:
        assert col in gdf.columns
