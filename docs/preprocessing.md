# Dataset Preprocessing Guide

This document provides a technical overview of the preprocessing modules found in `src/preprocessing/`. These modules are responsible for transforming raw data into a standardized format suitable for the attribution pipeline. The entire workflow is orchestrated by the `src/inference/preprocess_excerpts.py` script.

Each preprocessing module generates a GeoParquet file in the `outputs/preprocessing/` directory. These files adhere to a common schema, including columns for geometry, start/end dates, and disturbance class, ensuring they can be consumed by the attribution phase.

---

## Preprocessing Modules

Below are the details of each dataset's preprocessing logic.

### Senf & Seidl Disturbance Maps

- **Module**: `src/preprocessing/senfseidl.py`
- **Function**: `process_senfseidl()`
- **Inputs**: 
    - Disturbance attribution raster (`...fire_wind_barkbeetle...tif`)
    - Disturbance year raster (`...disturbance_year...tif`) from `excerpts/raw/senfseidl/`
- **Steps**:
  1. Reads the year and cause raster files.
  2. Converts raster pixels with valid disturbance data into polygons for each year.
  3. Joins the year and cause information for each polygon.
  4. Maps the raw raster values to the final disturbance classes defined in `src/config/constants.py`. This includes handling non-injective mappings where one raw value can map to multiple final classes (e.g., 'Storm/Biotic').
  5. Filters the polygons to the specified year range.
  6. Saves the resulting GeoDataFrame to `outputs/preprocessing/senfseidl_processed.parquet`.

### Health Monitoring (HM)

- **Module**: `src/preprocessing/hm.py`
- **Function**: `process_hm()`
- **Input**: A single Parquet file from `excerpts/raw/hm/`.
- **Steps**:
  1. Loads the raw point data from the Parquet file.
  2. Filters events to the specified year range.
  3. Maps the detailed survey categories to the final, harmonized disturbance classes.
  4. Standardizes column names and data types.
  5. Saves the processed point data to `outputs/preprocessing/hm_processed.parquet`.

### Fire Polygons

- **Module**: `src/preprocessing/firepolygons.py`
- **Function**: `process_firepolygons()`
- **Inputs**:
    - Polygon data from GPKG files in `excerpts/raw/firepolygons_gpkg/`.
    - Attribute data from `excerpts/raw/FFUD_Inventory_Arthur_excerpt.csv`.
- **Steps**:
  1. Reads the polygon geometries from the GPKG files.
  2. Reads the disturbance attributes (like event dates) from the CSV file.
  3. Merges the geometries with their corresponding attributes.
  4. Filters events based on the specified year range.
  5. Assigns the 'Fire' class to all events.
  6. Saves the final polygons to `outputs/preprocessing/firepolygons_processed.parquet`.

### Combined Drought Indicator (CDI)

- **Module**: `src/preprocessing/cdi.py`
- **Function**: `process_cdi()`
- **Input**: A set of yearly CDI raster files (`.tif`) from `excerpts/raw/cdi/`.
- **Steps**:
  1. Iterates through the raster files, filtering for the specified year range.
  2. For each raster, it identifies pixels indicating drought stress (values corresponding to 'Alert' or 'Warning' conditions).
  3. Converts these pixels into polygons.
  4. Assigns the 'Drought' class and the corresponding year to the polygons.
  5. Merges the data from all processed rasters.
  6. Saves the drought polygons to `outputs/preprocessing/cdi_processed.parquet`.

### FORMS Clear-Cuts

- **Module**: `src/preprocessing/forms.py`
- **Function**: `process_forms()`
- **Input**: Raster files (`.tif`) representing potential clear-cuts from `excerpts/raw/forms/`.
- **Steps**:
  1. Processes each raster file within the specified year range.
  2. Identifies pixels that represent clear-cuts based on their value.
  3. Converts these pixels into polygons.
  4. Assigns the 'Anthropogenic' class and the corresponding year.
  5. Merges data from all rasters.
  6. Saves the clear-cut polygons to `outputs/preprocessing/forms_processed.parquet`.
