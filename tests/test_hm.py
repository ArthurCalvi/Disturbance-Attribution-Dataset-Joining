from src.preprocessing.hm import process_hm


def test_process_hm(tmp_path):
    gdf = process_hm('excerpts/raw/excerpt_health_monitoring.parquet', tmp_path/'out.parquet')
    assert gdf.crs.to_string() == 'EPSG:2154'
    expected = {
        'year',
        'geometry',
        'cause',
        'notes',
        'severity',
        'class',
        'essence',
        'tree_type',
        'dataset',
    }
    assert expected.issubset(gdf.columns)
