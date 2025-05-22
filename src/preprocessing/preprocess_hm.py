"""Health Monitoring survey preprocessing.

Source: French DSF health monitoring data (private).
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import geopandas as gpd
import pandas as pd
from .utils import ensure_columns


def preprocess_hm(input_path: str | Path, output_path: str | Path | None = None) -> gpd.GeoDataFrame:
    """Load and standardise the health monitoring points."""
    gdf = gpd.read_parquet(input_path) if str(input_path).endswith("parquet") else gpd.read_file(input_path)
    if "year" in gdf.columns and "start_date" not in gdf.columns:
        gdf["start_date"] = pd.to_datetime(gdf["year"].astype(int).astype(str) + "-01-01")
        gdf["end_date"] = pd.to_datetime(gdf["year"].astype(int).astype(str) + "-12-31")
    gdf = ensure_columns(gdf, "hm")
    if output_path:
        gdf.to_parquet(output_path)
    return gdf

