import time
import geopandas as gpd
import pytest

pytest.importorskip("hdbscan")

from src.attribution import Attribution


def _load(path: str, n: int = 20) -> gpd.GeoDataFrame:
    return gpd.read_parquet(path).head(n)


def test_attribution_pipeline():
    gdfs = {
        "cdi": _load(
            "excerpts/pre-processed/excerpt_simplified_CDI_2012_2023_EPSG2154_FR.parquet"
        ),
        "firepolygons": _load(
            "excerpts/pre-processed/excerpt_simplified_firepolygons_2017_2020_FR_EPSG2154.parquet"
        ),
        "hm": _load(
            "excerpts/pre-processed/excerpt_simplified_health-monitoring_2007-2023_EPSG2154_FR.parquet"
        ),
        "forms": _load(
            "excerpts/pre-processed/excerpt_simplified_FORMS_clearcut_2017_2020_EPSG2154.parquet"
        ),
        "senfseidl": _load(
            "excerpts/pre-processed/excerpt_simplified_SenfSeidl_joined_EPSG2154_FR.parquet"
        ),
    }
    attr = Attribution(gdfs)

    timings = {}
    start = time.perf_counter()
    attr.build_graph()
    timings["build_graph"] = time.perf_counter() - start

    start = time.perf_counter()
    attr.detect_communities()
    timings["detect_communities"] = time.perf_counter() - start

    start = time.perf_counter()
    attr.run_hdbscan()
    timings["run_hdbscan"] = time.perf_counter() - start

    start = time.perf_counter()
    result = attr.attribute()
    timings["attribute"] = time.perf_counter() - start

    assert attr.graph.number_of_nodes() == len(attr.data)
    assert "community_id" in attr.data.columns
    assert isinstance(result, dict)
    assert all(t > 0 for t in timings.values())
