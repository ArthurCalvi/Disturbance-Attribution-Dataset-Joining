"""Process FORMS height rasters to detect clear-cuts.

Original data path used in the notebook:
'/Users/arthurcalvi/Data/Disturbances_maps/FORMS/'
"""
from pathlib import Path
from typing import List
from datetime import datetime
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import numpy as np


def _difference_rasters(r1: str, r2: str) -> np.ndarray:
    """Return binary difference array between two rasters."""
    with rasterio.open(r1) as src1, rasterio.open(r2) as src2:
        arr = src1.read(1) - src2.read(1)
        arr[(src2.read(1) > 500)] = 32767
        arr[(arr < 500)] = 32767
        arr[(arr > 32767 - 5000)] = 32767
        arr[arr == 32767] = 0
        arr[arr > 0] = 1
        return arr.astype("uint8"), src1.transform, src1.crs


def _extract_year(path: str) -> str:
    """Extract the first 4-digit year found in a file name."""
    stem_parts = Path(path).stem.split("_")
    for part in stem_parts:
        if part.isdigit() and len(part) == 4:
            return part
    raise ValueError(f"Year not found in filename: {path}")


def process_forms(rasters: List[str], output_file: str) -> gpd.GeoDataFrame:
    """Compute differences between consecutive rasters and create polygons."""
    polygons = []
    rasters = sorted(rasters)
    for r1, r2 in zip(rasters[:-1], rasters[1:]):
        diff, transform, crs = _difference_rasters(r1, r2)
        for geom, val in shapes(diff, mask=diff > 0, transform=transform):
            geom = shape(geom)
            year1 = _extract_year(r1)
            year2 = _extract_year(r2)
            polygons.append({
                'geometry': geom,
                'start_date': datetime.strptime(f'09-{year1}', '%m-%Y'),
                'end_date': datetime.strptime(f'05-{year2}', '%m-%Y'),
            })
    if polygons:
        gdf = gpd.GeoDataFrame(polygons, geometry='geometry', crs=crs).to_crs('EPSG:2154')
    else:
        gdf = gpd.GeoDataFrame(
            columns=['geometry', 'start_date', 'end_date'],
            geometry='geometry',
            crs='EPSG:2154',
        )
    gdf['class'] = 'clear-cut'
    gdf['dataset'] = 'forms'
    gdf['year'] = gdf['start_date'].dt.year
    if output_file:
        gdf.to_parquet(output_file)
    return gdf
