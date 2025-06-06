# Project Progress

This file tracks the major steps and changes made to the repository.

## 2024-07-25

- **Refactor data excerpt creation scripts:**
    - Centralized bounding box definitions in `src/config/constants.py` under `EXCERPT_BOUNDING_BOXES`.
    - Added a 'default' and a 'vosges' bounding box.
    - This will allow for easier management and selection of areas of interest for data excerpting.
- **Refactor data excerpt creation scripts (`data_excerpt_creation/`):**
    - Modified all scripts to accept a `--bbox-name` command-line argument to select a bounding box from `constants.py`.
    - Removed all hardcoded bounding boxes from the scripts.
    - Removed all logic related to file size checking, warnings, and automated reduction of the bounding box or data sampling. The scripts will now create excerpts of any size based on the chosen bounding box.
    - Updated output filenames to include the name of the bounding box used (e.g., `excerpt_default_...`).
    - Standardized output directories for raster excerpts into their own subfolders within `excerpts/raw/`.
- **Created a master script for orchestration:**
    - Added `data_excerpt_creation/create_all_excerpts.py` to run all individual excerpt creation processes from a single command.
    - This script takes a `--bbox-name` argument to pass to all sub-scripts.
    - Refactored all individual excerpt scripts to make them importable and callable from this master script.
- **Improved Logging Output:**
    - Modified the master script to time each major excerpting step and log the duration.
    - Added a `--verbose` flag to the master script to control log output.
    - Demoted verbose, step-by-step logs in all individual scripts from `INFO` to `DEBUG` level.
    - The default output is now a clean, high-level summary, with detailed logs available via the `--verbose` flag.

## Summary

- Created raw data excerpts to facilitate testing.
- Refactored preprocessing notebooks into `src/preprocessing/` modules with
  unit tests.
- Implemented a new graph-based attribution pipeline in `src/attribution/` with
  tests.
- Next objective is to integrate these components through an `src/inference/`
  package.

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
    *   Added a section detailing the attribution process. 

# Project Progress Log

## Initial Setup
- Cloned repository.
- Analyzed existing `create_forms_raster_excerpt.py` script.

## Task: Modify script for multiple files and synchronized BBOX reduction
- **Goal:** Adapt the script to process a list of input GeoTIFF files. If any file requires its BBOX to be reduced to meet size constraints, this reduction (as a scaling factor) should be applied to a master BBOX, and all files should then be processed using this commonly scaled BBOX.

- **Plan:**
    1.  **Update Constants**:
        *   Change `FORMS_RASTER_FILE_PATH` to `FORMS_RASTER_FILE_PATHS` (list).
        *   Make output filenames dynamic based on input filenames.
    2.  **Refactor Core Logic with New Helper Functions**:
        *   `_get_raster_info(src_path)`: Fetches raster metadata (profile, CRS, nodata).
        *   `_crop_and_write_raster(...)`: Handles the actual cropping and writing of a raster excerpt using a given BBOX. It will manage temporary files for size checking and write the final output. It will also handle edge cases like no BBOX overlap or zero-dimension crops by creating minimal valid rasters.
    3.  **New Sizing Strategy Function `determine_min_bbox_and_scale_for_size(...)`**:
        *   This function will replace the core logic of the old `create_raster_excerpt_with_size_control`.
        *   It will take a single raster path, an initial BBOX for that raster (already in the raster's CRS), and size constraints.
        *   It will use `_crop_and_write_raster` with temporary file paths to perform iterative scaling (direct calculation then fallback attempts) if the initial BBOX excerpt is too large.
        *   It will return the final BBOX achieved *for that file* (in its CRS) and the *effective side scaling factor* that was applied to the initial BBOX it received to meet the constraint. If no scaling was needed, this factor is 1.0.
    4.  **Main Function `create_forms_raster_excerpts_synced_bbox(...)`**:
        *   This function will replace the old `create_forms_raster_excerpt()`.
        *   It will manage a two-pass process.
        *   **Pass 1: Determine Common BBOX Scaling**:
            *   Define a `master_initial_bbox_epsg2154` from `BBOX_COORDS_EPSG2154`.
            *   For each input file:
                *   Get its CRS and transform `master_initial_bbox_epsg2154` to this CRS.
                *   Call `determine_min_bbox_and_scale_for_size` with this transformed BBOX.
                *   Track the minimum `effective_side_scale_factor` required across all files. Let this be `overall_min_effective_side_scale`.
            *   If `overall_min_effective_side_scale < 1.0`, scale down the `master_initial_bbox_epsg2154` by this factor to get the `final_common_bbox_epsg2154`. Otherwise, `final_common_bbox_epsg2154` is the same as `master_initial_bbox_epsg2154`.
        *   **Pass 2: Create Final Excerpts**:
            *   For each input file:
                *   Transform `final_common_bbox_epsg2154` to the file's CRS.
                *   Call `_crop_and_write_raster` to generate the final excerpt using this (potentially reduced) common BBOX.
                *   Log the final size. If it still exceeds `HARD_MAX_SIZE_MB` (e.g., due to compression differences or if the common BBOX wasn't small enough for this specific file's content), log a critical warning.
    5.  **Utilities**:
        *   Use `tempfile` module for managing temporary files/directories needed during the size determination pass.
        *   Update logging throughout.

## Current Refactoring Status

- Preprocessing notebooks have been migrated into the `src/preprocessing/` package with unit tests validating each dataset handler.
- A new `src/attribution/` package implements the updated graph-based pipeline described in `ATTRIBUTION.md` and includes its own tests.
- **Next mission:** create an `src/inference/` directory that orchestrates the full workflow from preprocessing through attribution.

## Task: Create `src/inference/preprocess_excerpts.py`
- Create a script to preprocess all raw excerpts from `excerpts/raw/` using the modules in `src/preprocessing/`.
- Save outputs to `outputs/preprocessing/`.
- Ensure all datasets (`cdi`, `hm`, `SenfSeidl`, `firepolygons`, `forms`) are processed.
- Script created and ran successfully.

## Task: Create `src/inference/perform_attribution.py`
- Create a script to load preprocessed data from `outputs/preprocessing/`.
- Use the `Attribution` class from `src/attribution/` to perform graph building, community detection, HDBSCAN clustering, and final attribution.
- Save the attribution results (e.g., the final GeoDataFrame with cluster/community info).
- Encountered `TypeError` due to mixed timezone-aware and naive datetimes during graph building.
- Modified `src/attribution/pipeline.py` to standardize date columns to naive UTC in `_prepare_data` method.
- Added record count logging to `src/inference/preprocess_excerpts.py` for better visibility of preprocessing output sizes.

## Task: Investigate High Event Count from Senf & Seidl
- User concerned about ~500k events from Senf & Seidl preprocessing.
- Plan to create a QC script (`src/qc/check_preprocessing.py`) to:
    - Generate a visual plot of preprocessed polygons.
    - Output summary statistics (e.g., area distribution).
- Modify `preprocess_excerpts.py` to run this QC script for each dataset.
- Discuss potential use of `geopandas.dissolve` to consolidate events if QC indicates fragmentation.

## Task: Adjust Attribution Parameters & Re-run
- User concerned about large default spatial parameters in the attribution pipeline (6000m radius, 1000m half-life).
- Reduced `spatial_half_life` to 300m, candidate search radius to 1500m, and hard distance cutoff to 1500m in `src/attribution/pipeline.py` for a more realistic starting point, especially given Senf & Seidl data granularity.

## Task: Refine Senf & Seidl Preprocessing for Better Consolidation
- Observed that the high number of Senf & Seidl events (~500k) is likely due to pixel-level vectorization.
- Refactored `src/preprocessing/senfseidl.py` based on `Process_SenfSeidlmap.ipynb` logic:
    - Independent vectorization of year and cause rasters.
    - Spatial join between year and cause polygons.
    - Dissolve by year and cause to merge contiguous/overlapping events.
    - Explode MultiPolygons to ensure one feature per distinct polygon.
- This aims to reduce feature count and better represent distinct disturbance events.
- Corrected `TypeError` in `preprocess_excerpts.py` call to `senfseidl.process_senfseidl` due to renamed keyword arguments.
- Corrected `NameError: name 'pd' is not defined` in `src/preprocessing/senfseidl.py` by adding `import pandas as pd`.

### Documentation

*   **Created `excerpts/excerpts_creation_summary.md`:**
    *   Added a brief markdown file summarizing the excerpt creation process for the Senf & Seidl rasters and the Health Monitoring vector data.
    *   This summary notes how excerpts were made and any changes in file type from the original, which is relevant for understanding preprocessing needs.
*   **Updated `excerpts/excerpts_creation_summary.md`:**
    *   Added a section detailing the excerpt creation for the Fire Polygons data (both GPKG geometries and CSV attributes).
    *   Added a new section detailing the CDI raster excerpt creation process, noting the cropping, data type optimization (`uint8`), LZW compression, and that the output remains as individual GeoTIFF files (one per time step) but smaller and clipped.
    *   Added a section detailing the FORMS raster excerpt creation process.
    *   Added a section detailing the attribution process. 

## Task: Implement Date Range Filtering
- Added `--start-year` and `--end-year` CLI arguments to `src/inference/preprocess_excerpts.py`.
- Updated `src/preprocessing/senfseidl.py` to accept `start_year` and `end_year` and filter data accordingly.
- Updated `src/preprocessing/cdi.py` to accept `start_year` and `end_year` and filter data accordingly.
- Updated `src/inference/preprocess_excerpts.py` to pass year arguments to `process_senfseidl` and `process_cdi`.

## Task: Investigate Missing "Anthropogenic" Class from Senf & Seidl
- User noted the absence of "Anthropogenic" disturbances from Senf & Seidl outputs.
- **Initial thought**: Issue with Senf & Seidl preprocessing or class mapping.
    - Reviewed `src/config/constants.py`: `RAW_TO_FINAL_TARGET_MAPPINGS['senfseidl']` does not currently have a mapping that results in 'Anthropogenic'.
    - Reviewed `src/preprocessing/senfseidl.py`: The `SENFSEIDL_CODE_TO_RAW_CAUSE` map (`{1: 'Storm,Biotic', 2: 'Fire', 3: 'Other'}`) does not include a code for anthropogenic activities.
- **Clarification**: User expects Senf & Seidl events to be *attributed* as 'Anthropogenic' if they cluster with 'Anthropogenic' events from other datasets (e.g., FORMS), even if Senf & Seidl itself doesn't have an original 'Anthropogenic' class.
- **Current Investigation**: Reviewing the `attribute` method in `src/attribution/pipeline.py`.
    - The method calculates weighted votes for classes from all datasets within a cluster.
    - It adds a 'self-vote' for the Senf & Seidl event's original class.
    - Probabilities are then normalized.
- **Hypotheses for missing 'Anthropogenic' attribution for Senf & Seidl events**:
    1.  Relevant Senf & Seidl events are not clustering with 'Anthropogenic' events from FORMS (or other datasets).
    2.  Senf & Seidl's original class (which gets a self-vote) combined with votes from other non-anthropogenic classes in the cluster outweighs any 'Anthropogenic' votes from FORMS.
    3.  Spatio-temporal parameters for graph building/clustering might be too restrictive, preventing linkage between Senf & Seidl and FORMS events.
- **Next Steps**: User to investigate cluster compositions of specific Senf & Seidl events to see if they are co-clustered with FORMS 'Anthropogenic' events. Based on findings, parameters or attribution logic might need adjustment.

## Task: Remap Senf & Seidl 'Other' Class to 'Anthropogenic'
- Based on user feedback that Senf & Seidl's 'Other' class (cause code 3) often represents anthropogenic disturbances in reality.
- Modified `src/config/constants.py` in `RAW_TO_FINAL_TARGET_MAPPINGS['senfseidl']` to map `'Other'` to `'Anthropogenic'` (previously `'Unknown'`).
- This will result in Senf & Seidl events with cause code 3 being directly classified as 'Anthropogenic' during preprocessing.
- This change should increase the prevalence of 'Anthropogenic' attributes for Senf & Seidl events, both directly and through the self-vote mechanism in the attribution pipeline.
- User will re-run preprocessing and attribution to observe the impact.

## 2025-06-05

*   **Task Started:** Adapt preprocessing scripts and QC to handle list-based `RAW_TO_FINAL_TARGET_MAPPINGS`.
    *   Identified errors: `TypeError: unhashable type: 'numpy.ndarray'` in QC, `ValueError: Length of values (1) does not match length of index` in `cdi.py` and `forms.py`.
    *   Plan: Inspect and modify all scripts in `src/preprocessing`, and `src/qc/check_preprocessing.py`.
*   **Task Completed:** 
    *   Updated `src/preprocessing/cdi.py`, `src/preprocessing/forms.py`, `src/preprocessing/firepolygons.py` to correctly assign list-based classes using list comprehensions and robust default handling.
    *   Updated `src/preprocessing/senfseidl.py` and `src/preprocessing/hm.py` to ensure their class mapping logic consistently produces lists and handles defaults to `['Unknown']` correctly.
    *   Updated `src/qc/check_preprocessing.py` to create a stringified `'class_display'` column from list-based classes for use in `nunique()` and `plot()`, resolving the `TypeError`.
    *   Updated `src/inference/preprocess_excerpts.py` to use a similar stringification method for `value_counts()` in all class distribution logging sections. 

## Task: Enhance QC Plotting with GIS-Style Cartography

- **Goal:** Improve the visual quality and information content of the Quality Control (QC) plots generated by `src/qc/plot_attribution_results.py` for the final disturbance attribution results, incorporating professional GIS-style cartographic features.

- **Enhancements Implemented:**
    - **`src/config/constants.py` Updates:**
        - Revised `DISTURBANCE_CLASS_COLORS` with a professional, colorblind-friendly palette inspired by ColorBrewer, suitable for cartographic representation.
        - Added comments to clarify the choice of colors and their intended use for various disturbance classes, including combined classes.

    - **`src/qc/plot_attribution_results.py` Overhaul:**
        - **General Plotting Enhancements:**
            - Plots are now saved at a higher resolution (300 DPI) for publication quality.
            - Figure and axes backgrounds are styled for better visual appeal (e.g., `facecolor='white'` for figure, light gray for axes).
            - Logging messages were improved for clarity during plot generation.
            - Output filenames for plots now include a `_pro.png` suffix (e.g., `spatial_attributed_classes_senfseidl_pro.png`) to distinguish the enhanced versions.
            - Defensive coding: Added checks for empty GeoDataFrames or GeoDataFrames with all empty geometries before attempting to plot.
            - Active geometry column is explicitly set in `main()` using `gdf.set_geometry('geometry')`.
            - Passed copies of GeoDataFrames (`.copy()`) to plotting functions from `main()` to prevent `SettingWithCopyWarning`.

        - **Spatial Distribution Plots (`plot_spatial_distribution`):**
            - **North Arrow:** A professional north arrow (`<-_>fancy` style) is now added to each spatial plot using `add_north_arrow()`.
            - **Dynamic Scale Bar:** A dynamic scale bar (`add_scale_bar()`) is added, which adjusts its length and label (meters or kilometers) based on the extent of the data in the plot. It assumes the CRS is in meters.
            - **Axis Styling:** Axes are formatted with `format_axes_professional()` including bold labels for "Easting" and "Northing", grid lines (subtle, dashed gray), and scientific notation for tick labels (using `useMathText=True` where available).
            - **Legend:** The legend (`create_professional_legend()`) is significantly improved. For categorical data (like attributed classes), it now displays:
                - Class name.
                - Count of events for that class.
                - Percentage of total events for that class.
                - Legend is placed outside the plot area to avoid overlap, with a styled frame.
            - **Color Handling:** Uses the `DISTURBANCE_CLASS_COLORS`. For categorical plots, it directly maps these colors to the polygons. For continuous data (like `attribution_confidence_final`), it uses the 'RdYlBu_r' colormap.
            - **Layout:** `plt.tight_layout()` is used with adjustments to accommodate the external legend and other elements.
            - Polygon `edgecolor` set to 'black' with a fine `linewidth` for better definition.

        - **Class Distribution Pie/Bar Charts (`plot_class_pie_chart`):**
            - **Dual Plot:** The function now generates a figure with two subplots:
                1.  A styled pie chart (donut style with `wedgeprops=dict(width=0.4)`).
                2.  A complementary bar chart showing the absolute counts for each class.
            - **Pie Chart Styling:**
                - Percentages are displayed inside the wedges, in bold white text for better contrast.
                - Labels are removed from the pie chart itself; information is moved to a shared legend.
            - **Bar Chart Styling:**
                - Bars use the `DISTURBANCE_CLASS_COLORS`.
                - Absolute counts are displayed on top of each bar.
                - X-axis labels (class names) are rotated for readability.
                - Y-axis shows "Number of Events" with a grid.
            - **Shared Legend:** A single, detailed legend is created for both the pie and bar chart, showing class name, absolute count, and percentage. It's placed to the side of the subplots.
            - **Overall Title:** A main title is set for the entire figure using `fig.suptitle()`.

- **Testing Note:** User needs to ensure the script is run in the correct Python virtual environment with compatible `numpy` and `matplotlib` versions to avoid import errors previously encountered. 

## Task: Fix Import Errors and Enhance Attribution Logic

- **Goal:** Fix import errors in QC plotting script and improve attribution logic to handle isolated Senf & Seidl events with ambiguous classes.

- **Fixes Implemented:**
    - **`src/qc/plot_attribution_results.py` Import Fix:**
        - Added dynamic path resolution to include project root in `sys.path` to resolve `ModuleNotFoundError: No module named 'src'`.
        - Fixed arrow style from non-existent `'<-_>fancy'` to standard `'->'` with better styling.
        - Fixed deprecated `pd.api.types.is_categorical_dtype()` to `isinstance(dtype, CategoricalDtype)`.
        - Removed conflicting `column` parameter when using `color` in `gdf.plot()`.
        - Removed polygon edges (`edgecolor='none'`) in all spatial plots for cleaner visualization.

    - **`src/attribution/pipeline.py` Attribution Logic Enhancement:**
        - Added fallback mechanism for isolated Senf & Seidl events with ambiguous classes.
        - Defined ambiguous class mappings:
            - `'Other'` → `['Anthropogenic', 'Unknown', 'Drought']`
            - `'Storm,Biotic'` → `['Storm', 'Biotic']`
        - When a Senf & Seidl event is isolated (no cluster votes) and has an ambiguous class, it's assigned to "Unknown" with 100% confidence.
        - Added statistics tracking for:
            - Total isolated Senf & Seidl events
            - Isolated events with ambiguous classes

    - **`src/inference/perform_attribution.py` Community Analysis:**
        - Added analysis after community detection to identify communities containing only Senf & Seidl polygons.
        - Reports:
            - Number of Senf&Seidl-only communities
            - Total events in these communities
            - Count of events with ambiguous classes in these communities

- **Rationale:** The previous issue where Drought appeared as the dominant class (67.2%) was due to isolated Senf & Seidl events with "Other" class getting equal votes for ['Anthropogenic', 'Unknown', 'Drought'], with Drought winning alphabetically. The new logic ensures these ambiguous cases are properly assigned to "Unknown" instead. 