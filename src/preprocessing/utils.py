"""Utility functions for preprocessing modules."""
from __future__ import annotations

from typing import Iterable
import pandas as pd
import geopandas as gpd

REQUIRED_COLUMNS = [
    "geometry",
    "start_date",
    "end_date",
    "class",
    "dataset",
    "year",
]

OPTIONAL_COLUMNS = ["cause", "tree_type", "essence"]


def ensure_columns(gdf: gpd.GeoDataFrame, dataset_name: str) -> gpd.GeoDataFrame:
    """Ensure common columns exist and CRS is EPSG:2154."""
    gdf = gdf.copy()
    if gdf.crs is None or gdf.crs.to_epsg() != 2154:
        gdf = gdf.to_crs(2154)
    gdf["dataset"] = dataset_name
    if "year" not in gdf.columns:
        gdf["year"] = gdf["start_date"].dt.year
    for col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        if col not in gdf.columns:
            gdf[col] = None
    gdf = gdf[REQUIRED_COLUMNS + OPTIONAL_COLUMNS]
    return gdf

