"""FORMS clear-cut preprocessing.

Source: https://console.cloud.google.com/storage/browser/height_maps
"""
from __future__ import annotations

from pathlib import Path
import geopandas as gpd
import pandas as pd
from .utils import ensure_columns


def preprocess_forms(input_path: str | Path, output_path: str | Path | None = None) -> gpd.GeoDataFrame:
    """Load and standardise FORMS clear-cut polygons."""
    gdf = gpd.read_parquet(input_path) if str(input_path).endswith("parquet") else gpd.read_file(input_path)
    if "start_date" in gdf.columns:
        gdf["start_date"] = pd.to_datetime(gdf["start_date"])
    if "end_date" in gdf.columns:
        gdf["end_date"] = pd.to_datetime(gdf["end_date"])
    gdf = ensure_columns(gdf, "forms")
    if output_path:
        gdf.to_parquet(output_path)
    return gdf

