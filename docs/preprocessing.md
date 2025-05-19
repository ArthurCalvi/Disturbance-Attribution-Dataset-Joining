# Dataset Preprocessing Guide

This guide details how each raw dataset is prepared before attribution. The notebooks live in `process-datasets/` and produce simplified `.parquet` files inside `data/processed_datasets/`. These paths correspond to the entries in `join-datasets/constants.py`.

## Workflow Overview
1. Execute the preprocessing notebooks in the order listed below.
2. Verify the output files exist at the expected locations.
3. Proceed to the attribution stage once all processed datasets are available.

## Preprocessing Steps per Dataset

### Senf & Seidl Disturbance Maps
- **Notebook**: `Process_SenfSeidlmap.ipynb`
- **Inputs**: Annual raster tiles describing disturbance year and type.
- **Steps**:
  1. Convert raster tiles to polygons.
  2. Spatially join the attribution and year layers.
  3. Merge polygons by disturbance type and year.
  4. Save yearly polygons of fire, storm/beetle and other events.

### DFDE Records
- **Notebook**: `Process_DFDE_FR.ipynb`
- **Inputs**: DFDE CSV/Excel export containing disturbance reports.
- **Steps**:
  1. Geocode administrative names using a GeoParsing tool and OpenStreetMap.
  2. Clean species names and translate when necessary.
  3. Create polygons for each reported event with associated dates and classes.

### French National Forest Inventory (NFI)
- **Notebooks**: `Process_NFI_FR.ipynb` and `PostProcess_NFI.ipynb`
- **Steps**:
  1. Filter plots reporting incidents or visible management activity.
  2. Convert the coordinates to EPSG:2154 and build a GeoDataFrame.
  3. Map the `Incident` field to the six final disturbance classes.
  4. Export points with disturbance class and start/end dates.

### Health Monitoring Survey (HMS)
- **Notebook**: `Process_health-monitoring_FR.ipynb`
- **Steps**:
  1. Keep observations with medium or higher severity.
  2. Map detailed survey categories to the six final classes.
  3. Output annual point data with disturbance type and species.

### Combined Drought Indicator (CDI)
- **Notebook**: `Process_cdi.ipynb`
- **Steps**:
  1. Select pixels repeatedly flagged "Alert" followed by "Temporary Recovery".
  2. Convert the selected pixels to polygons for each year.
  3. Save yearly drought polygons.

### Fire Polygons from Sentinel‑2
- **Notebook**: `Process_firepolygons.ipynb`
- **Steps**:
  1. Read daily burned area polygons derived from Sentinel‑2 imagery.
  2. Export the polygons as daily records.

### FORMS Clear‑Cuts
- **Notebook**: `Preprocess_FORMS.ipynb`
- **Steps**:
  1. Compute canopy height difference using Sentinel‑2 and GEDI inputs.
  2. Keep areas with a drop larger than 5 m and size above 0.5 ha.
  3. Filter polygons by geometric shape to retain likely clear‑cuts.
  4. Write yearly clear‑cut polygons.

### BDIFF Fire Records
- **Notebook**: `process_bdiff.ipynb`
- **Steps**:
  1. Geocode administrative units mentioned in BDIFF.
  2. Keep only fires larger than 0.5 ha.
  3. Save polygons with burned area and alert date.

### BD Forêt Tree Species Map
- **Notebook**: `Process_BDFORET.ipynb`
- **Steps**:
  1. Standardise polygon resolution to 30 m.
  2. Translate tree species to English names.
  3. Export polygons for use in later joins.

All notebooks output simplified `.parquet` files used by the attribution module. Further methodological context is described in the `draft_article` document.
