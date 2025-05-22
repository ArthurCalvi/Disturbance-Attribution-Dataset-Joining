## Raw Data Excerpt Creation Summary

This document outlines how raw data excerpts were created for specific datasets, noting any changes from the original file format that might impact subsequent preprocessing steps.

All excerpts are cropped to the following BBOX: (307783.0822, 6340505.4366, 469246.8845, 6419190.9011), EPSG:2154

### Senf & Seidl Raster Excerpts (`create_senfseidl_raster_excerpts.py`)

Excerpts for the Senf & Seidl disturbance cause and year maps were created by cropping the original GeoTIFF files to a predefined bounding box (EPSG:2154, reprojected to the raster's native CRS for cropping). To reduce file size, the data types were optimized (e.g., to `uint8` or `uint16` where appropriate) and LZW compression was applied; the excerpts remain GeoTIFF files but with potentially different data types and are significantly smaller than the originals.

### Health Monitoring Vector Excerpt (`create_healthmonitoring_vector_excerpt.py`)

An excerpt for the Health Monitoring data (originally an Excel `.xlsx` file) was created by reading the relevant sheet, converting longitude/latitude to points (initial CRS EPSG:4326), reprojecting to EPSG:2154, and clipping to the standard BBOX. The output is a Parquet file (`.parquet`) stored in `excerpts/raw_vector/`, which is a change from the original Excel format; it includes robust error handling for Parquet data type conversions and a fallback to GeoPackage (`.gpkg`) if needed, plus file size management via sampling if the clipped data exceeds 5MB.

### Fire Polygons Vector & Tabular Excerpts (`create_firepolygons_excerpts.py`)

Excerpts for the Fire Polygons dataset involve two parts: individual fire polygon GeoPackage files and a main attribute CSV file. The script copies each original GPKG file from `/Users/arthurcalvi/Data/Disturbances_maps/FirePolygons/FFUD_Arthur/Fire_folder/`, reprojects it to EPSG:2154, clips it to the standard BBOX, and saves the (potentially empty) result as a new GPKG in `excerpts/raw_vector/firepolygons_gpkg/`. The main CSV attribute file (`FFUD_Inventory_Arthur.csv`) is copied directly to `excerpts/raw_tabular/`. The excerpts are therefore in the same format as the originals (GPKG and CSV), but the GPKGs are clipped. Preprocessing will later need to combine these clipped GPKG files and filter/join the CSV data accordingly.

### Combined Drought Indicator (CDI) Raster Excerpts (`create_cdi_raster_excerpts.py`)

Excerpts for the Combined Drought Indicator (CDI) dataset, which consists of multiple GeoTIFF files (e.g., `cdi_yyyyMMdd.tif` from `/Users/arthurcalvi/Data/Disturbances_maps/Copernicus_CDI/CDI_2012_2023/`), are created by processing each input GeoTIFF. Each raster is cropped to the standard BBOX (EPSG:2154, reprojected to the raster's native CRS, expected to be EPSG:3035), its data type is optimized to `uint8` (as CDI values are categorical and small), and LZW compression is applied. The processed excerpts are saved as individual GeoTIFF files in `excerpts/raw_raster/cdi/`, maintaining the GeoTIFF format but with optimized data types and significantly reduced file sizes.

### FORMS Forest Height Raster Excerpt (`create_forms_raster_excerpt.py`)

An excerpt for the FORMS forest height data, specifically from the `Height_mavg_2023.tif` file, is created by cropping the original GeoTIFF to the standard bounding box (EPSG:2154, with reprojection of the BBOX to the raster's native CRS if needed). To reduce file size, the data type is optimized to `int16` (assuming height values are in cm, requiring verification from the raw file) and LZW compression is applied. The excerpt remains a GeoTIFF file, saved in `excerpts/raw_raster/forms/`, but with a potentially different data type and a smaller spatial extent. 