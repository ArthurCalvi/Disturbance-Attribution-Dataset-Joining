from src.preprocessing.senfseidl import process_senfseidl


def test_process_senfseidl(tmp_path):
    gdf = process_senfseidl('excerpts/raw/excerpt_fire_wind_barkbeetle_france.tif',
                             'excerpts/raw/excerpt_disturbance_year_1986-2020_france.tif',
                             tmp_path/'out.parquet')
    assert gdf.crs.to_string() == 'EPSG:2154'
    expected = {'year','geometry','class','dataset'}
    assert expected.issubset(gdf.columns)
