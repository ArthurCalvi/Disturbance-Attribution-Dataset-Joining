from src.preprocessing.forms import process_forms


def test_process_forms(tmp_path):
    gdf = process_forms([
        'excerpts/raw/excerpt_Height_mavg_2022.tif',
        'excerpts/raw/excerpt_forms_height_mavg_2023.tif',
    ], tmp_path / 'out.parquet')
    assert gdf.crs.to_string() == 'EPSG:2154'
    expected = {'geometry', 'start_date', 'end_date', 'class', 'dataset', 'year'}
    assert expected.issubset(gdf.columns)
