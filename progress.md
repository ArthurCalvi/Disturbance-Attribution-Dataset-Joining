# Project Progress

## Initial Setup
- Understood the goal: Extract BBOX-defined excerpts from multiple spatial Parquet datasets.
- Identified target datasets from `join-datasets/constants.py`.
- Defined the BBOX for extraction: (307783.0822, 6340505.4366, 469246.8845, 6419190.9011) in EPSG:2154.
- Planned to create a new script in a dedicated folder for this task.

## Implementation
- Created `data_excerpt_creation/create_data_excerpts.py` script.
- The script defines the BBOX and target CRS (EPSG:2154).
- It dynamically loads `loading_dict` from `join-datasets/constants.py` to find dataset paths.
  - Note: The method for loading `loading_dict` by parsing the file content is fragile. A more robust approach would be to ensure `constants.py` is importable or use `ast.literal_eval`.
- It iterates through each dataset:
    - Constructs absolute paths for input Parquet files and output excerpt files.
    - Creates an output directory `data/excerpts/` (relative to `join-datasets/` parent directory).
    - Loads the Parquet file into a GeoDataFrame.
    - Ensures the GeoDataFrame is in the target CRS, reprojecting if necessary.
    - Clips the GeoDataFrame using the BBOX (`gdf.geometry.intersects(bbox_geometry)`).
    - Saves the clipped excerpt as a Parquet file.
    - Logs information, warnings (e.g., if excerpt > 5MB), and errors throughout the process.
    - Skips datasets if files are not found or if other errors occur during loading/processing.
- Added a summary log for successful and failed/skipped excerpts.

## Progress Log

### Setup and Initial Analysis

*   Reviewed project structure and refactoring goals from `AGENTS.md`.
*   Identified key datasets for preprocessing: `cdi`, `hm`, `SenfSeidl`, `firepolygons`, `forms`.
*   Noted the desired output structure: `excerpts/`, `src/`, `tests/`, `results/`.
*   Examined `create_data_excerpts.py` to understand existing excerpt creation logic for vector data.
*   Analyzed `Process_SenfSeidlmap.ipynb` to identify raw raster inputs and their initial CRS.

### Excerpt Creation - Raw Data

*   **Created `data_excerpt_creation/create_raster_excerpts.py`:**
    *   This script focuses on creating excerpts from the two raw raster files used in `Process_SenfSeidlmap.ipynb`:
        *   `../data/SenfSeidl_maps/fire_wind_barkbeetle_france.tif` (Cause map)
        *   `../data/SenfSeidl_maps/france/disturbance_year_1986-2020_france.tif` (Year map)
    *   Uses the same bounding box (EPSG:2154) as `create_data_excerpts.py` for consistency:
        *   `BBOX_COORDS_EPSG2154 = (307783.0822, 6340505.4366, 469246.8845, 6419190.9011)`
    *   The script reprojects this bounding box to the native CRS of each raster before cropping.
    *   The output raster excerpts are saved in `excerpts/raw/` and retain their original CRS.
    *   Includes logging for the process.
    *   Implements file size checks:
        *   Warning if an excerpt > 5MB.
        *   Critical warning if an excerpt > 10MB.
    *   Note: Automatic resampling for raster file size reduction is not yet implemented due to the complexity of defining a "random sampling of rows" for rasters. This can be addressed later by considering resolution changes or other methods if large files are produced.

*   **Created `data_excerpt_creation/create_healthmonitoring_excerpt.py`:**
    *   This script processes the external Excel file: `veille sanitaire DSF 2007_2023.xlsx`.
    *   It reads the specified sheet (`signalement0`, header on 3rd row).
    *   Converts Longitude/Latitude to points (initial CRS EPSG:4326).
    *   Reprojects the GeoDataFrame to EPSG:2154.
    *   Clips the data to the standard BBOX `(307783.0822, 6340505.4366, 469246.8845, 6419190.9011)`.
    *   Saves the output as `excerpts/raw_vector/excerpt_health_monitoring.parquet`.
    *   Includes logging and file size management (sampling if >5MB to target <9.5MB, with warnings for >5MB and critical for >10MB final size).
*   **Decision on Health Monitoring Data:** The `create_healthmonitoring_excerpt.py` script encountered runtime errors (numpy dtype incompatibility and Parquet conversion issues for string data). Given the relatively small size of the source Excel file (`veille sanitaire DSF 2007_2023.xlsx`), the decision has been made to manually include this raw file directly within the `excerpts/raw_data/health_monitoring/` directory for now, instead of programmatically generating a clipped excerpt. The script `data_excerpt_creation/create_healthmonitoring_excerpt.py` will not be used at this time.

*   **Re-created `data_excerpt_creation/create_healthmonitoring_vector_excerpt.py`:**
    *   Due to the Excel file size (30.4MB) being larger than initially thought, and to resolve previous Parquet export errors, this new script was created.
    *   It reads the `veille sanitaire DSF 2007_2023.xlsx` (sheet `signalement0`, header row 2).
    *   Uses corrected column names `Latitude` and `Longitude` for geometry creation (initial CRS EPSG:4326).
    *   Explicitly converts `Essence dominante`, `Essence concernée`, and `Essence regroupée (ess. concernée)` to string type based on observed errors.
    *   Implements a robust Parquet saving strategy: 
        1.  Attempts initial Parquet save.
        2.  If a `pyarrow.lib.ArrowInvalid` (type error) occurs, it attempts to convert all remaining `object` dtype columns to `string` and retries Parquet save.
        3.  If Parquet still fails, it falls back to saving as GeoPackage.
    *   Reprojects to EPSG:2154 and clips to the standard BBOX.
    *   Saves as `excerpts/raw/excerpt_health_monitoring.parquet` (or `.gpkg` on fallback).
    *   Includes file size management (sampling if >5MB to target <9.5MB, with warnings for >5MB and critical for >10MB final size).

*   **Analyzed `process-datasets/Process_health-monitoring_FR.ipynb`:**
    *   This notebook primarily processes an Excel file: `veille sanitaire DSF 2007_2023.xlsx` (path appears external to the project: `/Users/arthurcalvi/Data/Disturbances_maps/Thierry Belouard & DSF/Veille_sanitaire/`).
    *   It converts data from this Excel file into a GeoDataFrame (initially EPSG:4326) based on longitude/latitude columns.
    *   No direct raw raster file inputs were identified in this notebook that would require a similar raster excerpt creation process as SenfSeidl.
    *   Next step will be to clarify accessibility of the Excel file and determine if a vector excerpt (e.g., Parquet from the processed Excel data clipped to BBOX) is needed.

*   **Analyzed `process-datasets/Process_firepolygons.ipynb`:**
    *   This notebook processes fire polygon data from two main sources:
        1.  A CSV file with fire event attributes: `/Users/arthurcalvi/Data/Disturbances_maps/FirePolygons/FFUD_Arthur/FFUD_Inventory_Arthur.csv`.
        2.  A folder of GPKG files containing individual fire polygon geometries: `/Users/arthurcalvi/Data/Disturbances_maps/FirePolygons/FFUD_Arthur/Fire_folder/`.
    *   The notebook reads the CSV, then iterates through the GPKG files, reprojects them to EPSG:2154, and concatenates them.
    *   It merges the CSV attributes with the polygon geometries using a UID.
    *   It performs further data enrichment by spatially joining with a `BDFORET` Parquet file and later merges with a `bdiff` Parquet file.
    *   The final output is a merged Parquet file.
    *   For raw excerpt creation, we will need to process both the input CSV and the individual GPKG files.

*   **Created `data_excerpt_creation/create_firepolygons_excerpts.py`:**
    *   This script handles the raw data for the Fire Polygons dataset.
    *   **GPKG Processing:** It iterates through individual GPKG files located in `/Users/arthurcalvi/Data/Disturbances_maps/FirePolygons/FFUD_Arthur/Fire_folder/`.
        *   Each GPKG is read, reprojected to EPSG:2154 (target CRS), and clipped to the standard BBOX.
        *   Clipped non-empty geometries are saved as new GPKG files in `excerpts/raw_vector/firepolygons_gpkg/`.
        *   Includes logging and file size warnings for output GPKGs.
    *   **CSV Processing:** It copies the main attribute CSV file (`/Users/arthurcalvi/Data/Disturbances_maps/FirePolygons/FFUD_Arthur/FFUD_Inventory_Arthur.csv`) directly to `excerpts/raw_tabular/FFUD_Inventory_Arthur_excerpt.csv`.
        *   Includes logging and file size warnings for the copied CSV.

*   **Analyzed `process-datasets/Process_cdi.ipynb`:**
    *   This notebook processes Combined Drought Indicator (CDI) data, which are a series of GeoTIFF files (e.g., `cdi_yyyyMMdd.tif`), likely representing 10-day intervals.
    *   Input GeoTIFFs are sourced from `/Users/arthurcalvi/Data/Disturbances_maps/Copernicus_CDI/CDI_2012_2023/` (after an initial consolidation step from a parent directory).
    *   The notebook crops these rasters (EPSG:3035) to a France extent, then converts specific pixel values (e.g., value 4 for partial recovery) to vector polygons (GeoDataFrames), reprojects to EPSG:2154, and performs further clipping and drought event analysis.
    *   The final output is a Parquet file of drought polygons.
    *   For raw excerpt creation, we will process the individual input GeoTIFFs.

*   **Created `data_excerpt_creation/create_cdi_raster_excerpts.py`:**
    *   This script processes the raw CDI GeoTIFF files.
    *   It iterates through each `.tif` file in `/Users/arthurcalvi/Data/Disturbances_maps/Copernicus_CDI/CDI_2012_2023/`.
    *   Each raster is cropped to the standard BBOX (EPSG:2154, reprojected to the raster's CRS - expected EPSG:3035 - for the crop operation).
    *   The data type is optimized to `uint8`, and LZW compression is applied.
    *   Cropped rasters are saved as `excerpt_<original_filename>.tif` in `excerpts/raw_raster/cdi/`.
    *   Includes logging and file size warnings.

*   **Analyzed `process-datasets/Process_FORMS.ipynb`:**
    *   This notebook processes forest height GeoTIFFs (originally multiple yearly files, now focusing on `Height_mavg_2023.tif` from `/Users/arthurcalvi/Data/Disturbances_maps/FORMS/`).
    *   The primary processing involves calculating differences between annual height maps to detect clear-cuts, vectorizing these difference areas, and then filtering/attributing them (area, validity, dates, tree species via BDFORET intersection).
    *   The final output of the notebook is a Parquet file of clear-cut polygons.
    *   For raw excerpt creation from the single `Height_mavg_2023.tif`, a direct difference calculation isn't possible. Instead, the raw raster will be clipped and compressed.

*   **Created `data_excerpt_creation/create_forms_raster_excerpt.py` (and updated):**
    *   This script processes the raw FORMS `Height_mavg_2023.tif` raster file.
    *   It crops the GeoTIFF to the standard BBOX (EPSG:2154), reprojecting the BBOX to the raster's CRS for cropping if necessary.
    *   Applies LZW compression and attempts to optimize the data type to `int16` (assuming height in cm, nodata value handling included).
    *   **Iterative BBOX Reduction:** If the initial cropped excerpt exceeds 10MB, the script will iteratively reduce the BBOX size (by a scale factor, centered) and re-crop, up to a maximum number of attempts, to try and get the file size under 10MB.
    *   Saves the excerpt as `excerpts/raw_raster/forms/excerpt_forms_height_mavg_2023.tif`.
    *   Includes detailed logging for each reduction attempt and file size warnings.

### Documentation

*   **Created `excerpts/excerpts_creation_summary.md`:**
    *   Added a brief markdown file summarizing the excerpt creation process for the Senf & Seidl rasters and the Health Monitoring vector data.
    *   This summary notes how excerpts were made and any changes in file type from the original, which is relevant for understanding preprocessing needs.
*   **Updated `excerpts/excerpts_creation_summary.md`:**
    *   Added a section detailing the excerpt creation for the Fire Polygons data (both GPKG geometries and CSV attributes).
    *   Added a new section detailing the CDI raster excerpt creation process, noting the cropping, data type optimization (`uint8`), LZW compression, and that the output remains as individual GeoTIFF files (one per time step) but smaller and clipped.
    *   Added a section detailing the FORMS raster excerpt creation process. 