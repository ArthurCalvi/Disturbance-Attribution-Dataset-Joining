from src.preprocessing.firepolygons import process_firepolygons


def test_process_firepolygons(tmp_path):
    gdf = process_firepolygons('excerpts/raw/FFUD_Inventory_Arthur_excerpt.csv',
                               'excerpts/raw/firepolygons_gpkg', tmp_path/'out.parquet')
    assert gdf.crs.to_string() == 'EPSG:2154'
    expected = {'uuid','year','start_date','forest_area_m2','essence','name','geometry'}
    assert expected.issubset(gdf.columns)
