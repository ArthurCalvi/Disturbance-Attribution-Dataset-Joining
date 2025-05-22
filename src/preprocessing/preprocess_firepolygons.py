"""Fire polygons preprocessing.

Source: Sentinel-2 burned area polygons provided by Lilian Vallet (private).
"""
from __future__ import annotations

from pathlib import Path
import geopandas as gpd
import pandas as pd
from .utils import ensure_columns


def preprocess_firepolygons(input_path: str | Path, output_path: str | Path | None = None) -> gpd.GeoDataFrame:
    """Load and standardise fire polygon records."""
    gdf = gpd.read_parquet(input_path) if str(input_path).endswith("parquet") else gpd.read_file(input_path)
    if "start_date" in gdf.columns:
        gdf["start_date"] = pd.to_datetime(gdf["start_date"])
    if "end_date" not in gdf.columns:
        gdf["end_date"] = gdf["start_date"]
    gdf = ensure_columns(gdf, "firepolygons")
    if output_path:
        gdf.to_parquet(output_path)
    return gdf

