# Disturbance-Attribution-Dataset-Joining


## Required Dataset Columns

During preprocessing, ensure each output dataset provides at least the following columns so the joining algorithms work correctly:

- `geometry` – polygon or point geometry in EPSG:2154
- `start_date` and `end_date` – disturbance time range
- `class` – disturbance class
- `dataset` – dataset name (added automatically by the loader)
- `year` – fallback year used when detailed dates are missing

Additional fields improve similarity scoring when available:

- `cause` – precise disturbance cause
- `tree_type` – broad tree type
- `essence` – tree species

It is acceptable if some of these optional columns cannot be populated; the pipeline still runs but results may be less accurate.

## Disturbance Data Sources

- **Combined Drought Indicator** – https://edo.jrc.ec.europa.eu/

## Documentation
The preprocessing notebooks are described in [docs/preprocessing.md](docs/preprocessing.md).
The attribution workflow is explained in [docs/attribution.md](docs/attribution.md).
For further methodological context see `draft_article`.

## Tests
Run the unit tests with:
```bash
python -m unittest discover tests
```

# Forest Disturbance Attribution Pipeline

This repository contains a Python-based pipeline for attributing forest disturbances by integrating and analyzing data from multiple sources. It aims to provide a clearer understanding of disturbance events by combining the strengths of various datasets.

This document is intended to guide Jorunn, who will be continuing the development of this work, particularly with new Senf & Seidl data.

## Project Goal

The primary goal is to process various forest disturbance datasets, align them spatially and temporally, and then attribute a primary cause to events, especially those identified in a reference dataset (e.g., Senf & Seidl).

## Core Workflow

The pipeline operates in two main phases:

1.  **Preprocessing (`src/inference/preprocess_excerpts.py`):**
    *   Takes raw disturbance data (currently using excerpts in `excerpts/raw/` for development) for various sources (Senf & Seidl, CDI, Health Monitoring, Fire Polygons, FORMS).
    *   Each dataset has a dedicated module in `src/preprocessing/` (e.g., `src/preprocessing/senfseidl.py`).
    *   These modules clean, transform, and standardize the data, including harmonizing disturbance classes to a common set of `FINAL_TARGET_CLASSES` defined in `src/config/constants.py`.
    *   Outputs are GeoParquet files (e.g., `senfseidl_processed.parquet`) saved to `outputs/preprocessing/`.
    *   **Crucially, each preprocessed file must contain the following columns for the attribution phase:**
        *   `geometry`: Polygon or point geometry (EPSG:2154).
        *   `start_date`, `end_date`: Disturbance time range (datetime objects, timezone-naive UTC).
        *   `mid_date`: Calculated midpoint of `start_date` and `end_date`.
        *   `class`: Disturbance class, mapped to one of the `FINAL_TARGET_CLASSES` (e.g., 'Fire', 'Storm').
        *   `dataset`: Name of the source dataset (e.g., 'senfseidl', 'cdi').
        *   `year`: Integer year of the event, used as a fallback or for filtering.

2.  **Attribution (`src/inference/perform_attribution.py`):**
    *   Loads all preprocessed GeoParquet files from `outputs/preprocessing/`.
    *   Concatenates them into a single GeoDataFrame.
    *   Utilizes the `Attribution` class from `src/attribution/pipeline.py` to:
        1.  Build a spatio-temporal graph of all disturbance events.
        2.  Detect communities of related events using the Louvain algorithm.
        3.  Within each community, apply HDBSCAN to identify dense clusters of events.
        4.  Attribute a primary cause to events (currently focused on Senf & Seidl events) based on the classes of other clustered events from different datasets.
    *   Outputs include attributed event data (e.g., `attributed_events.parquet`) and cluster information, typically saved to `outputs/attribution/`.
    *   Quality control scripts in `src/qc/` (e.g., `plot_community_diagnostics.py`) are used to generate diagnostics for tuning.

## Key Scripts and Configuration

*   **Main Execution Scripts:**
    *   `src/inference/preprocess_excerpts.py`: Runs the preprocessing for all datasets.
    *   `src/inference/perform_attribution.py`: Runs the main attribution pipeline.
*   **Core Logic:**
    *   `src/attribution/pipeline.py`: Defines the `Attribution` class, containing methods for graph building, community detection (Louvain), clustering (HDBSCAN), and the core attribution logic.
    *   `src/preprocessing/`: Directory containing individual Python modules for each dataset's preprocessing steps (e.g., `senfseidl.py`, `cdi.py`).
*   **Configuration:**
    *   `src/config/constants.py`: Central file for defining:
        *   `FINAL_TARGET_CLASSES`: The list of standardized disturbance classes used throughout the pipeline.
        *   `RAW_TO_FINAL_TARGET_MAPPINGS`: Dictionaries mapping raw class labels from each input dataset to the `FINAL_TARGET_CLASSES`. This is crucial for class harmonization.
        *   (Potentially) Dataset reliability scores or other global parameters.
*   **Quality Control & Diagnostics:**
    *   `src/qc/`: Contains scripts for generating reports and visualizations to check the quality of preprocessing and help tune parameters (e.g., `check_preprocessing.py`, `plot_community_diagnostics.py`).

## Guide for Jorunn (Continuing Development)

This pipeline is designed to be flexible. Here are key areas you might want to focus on or modify, especially with your new Senf & Seidl map:

### 1. Senf & Seidl Preprocessing (`src/preprocessing/senfseidl.py`)

*   **Current Logic:** The current script processes Senf & Seidl year and cause rasters. It vectorizes polygons for each year/cause combination, then dissolves and explodes them to handle fragmented patches. Class assignment is based on the cause raster codes, mapped via `constants.py`.
*   **New Map Considerations (Multiple Disturbances per Pixel/Event):**
    *   Your new map might allow a single pixel or event to be associated with multiple disturbance types or intensities over time.
    *   The current preprocessing (`src/preprocessing/senfseidl.py`) and the downstream attribution logic (`src/attribution/pipeline.py`) assume one primary class per event polygon from Senf & Seidl.
    *   **This is a critical point:** If your new data has multiple disturbances per event, you will need to significantly adapt:
        1.  The Senf & Seidl preprocessing script to extract and represent this multi-label information. This might involve creating multiple records for the same geometry with different classes, or adding new columns to store probabilistic/multiple class assignments.
        2.  The `attribute` method in `src/attribution/pipeline.py` will need to be updated to correctly interpret and utilize this richer information from the Senf & Seidl data. The current voting mechanism might need to be rethought.

### 2. Class Mapping & Probabilistic Attribution (`src/config/constants.py`)

*   **Current Mapping:** `RAW_TO_FINAL_TARGET_MAPPINGS` in `constants.py` defines a direct (injective) mapping from raw dataset classes to one of the `FINAL_TARGET_CLASSES`. For example, the Senf & Seidl raw class 'Storm,Biotic' is currently mapped *only* to 'Biotic'.
*   **Potential Enhancement (Non-Injective/Probabilistic Mapping):**
    *   You might want to introduce a more nuanced mapping. For instance, a raw Senf & Seidl class like 'Storm,Biotic' could be mapped to multiple final classes with associated weights (e.g., 25% 'Storm' and 75% 'Biotic').
    *   To implement this:
        1.  You would need to modify the structure of `RAW_TO_FINAL_TARGET_MAPPINGS` in `constants.py` to support these probabilistic/weighted mappings. For example, instead of `raw_class: 'Final_Class'`, it could be `raw_class: {'Storm': 0.25, 'Biotic': 0.75}`.
        2.  The preprocessing scripts would need to be updated to create columns representing these probabilities (e.g., `prob_Storm`, `prob_Biotic`) instead of a single `class` column for such cases.
        3.  The `attribute` method in `src/attribution/pipeline.py` would need substantial changes to:
            *   Accept these probabilistic inputs from various datasets.
            *   Aggregate these probabilities during the voting/attribution process for Senf & Seidl events.

### 3. Louvain Community Detection Tuning (`src/attribution/pipeline.py`)

*   **Parameter:** The `louvain_resolution` parameter (passed to the `Attribution` class, typically in `perform_attribution.py`, and used in the `detect_communities` method) controls the granularity of the communities. Higher values lead to more, smaller communities; lower values lead to fewer, larger communities.
*   **Diagnostics:** Use the `src/qc/plot_community_diagnostics.py` script (called from `perform_attribution.py`) to visualize community size distributions. This will help you tune `louvain_resolution` to achieve communities of a size that makes sense for your analysis (e.g., ensuring communities are large enough for meaningful HDBSCAN clustering but not so large that distinct disturbance regimes are merged).

### 4. HDBSCAN Clustering Parameters (`src/attribution/pipeline.py`)

*   **Parameters:** Key HDBSCAN parameters like `min_cluster_size` and `min_samples` are set within the `_hdbscan_cluster` method in `src/attribution/pipeline.py`. The `min_cluster_size` is currently adapted based on the size of the Louvain community being processed.
*   **Tuning:** You may need to adjust these parameters based on the density and characteristics of events within communities. Experimentation might be needed to find optimal settings that capture meaningful disturbance clusters without excessive noise or over-segmentation.

### 5. Dataset Reliability / Weighting

*   **Concept:** The original design envisioned incorporating dataset reliability scores (e.g., `DCLASS_SCORE`, `ddataset_profile` in older versions of `constants.py` or passed to `Attribution`). These could be used to give more weight to data sources considered more reliable during the attribution process.
*   **Current Status & Enhancement:** Review the `attribute` method in `src/attribution/pipeline.py` to see how (or if) such scores are currently used in the voting logic.
*   **Tuning:** If implemented, these scores (potentially defined in `src/config/constants.py` or passed as arguments) would be another point for tuning the attribution outcomes. You could define scores per dataset or even per class within a dataset.

## Extending the Pipeline: Adding New Data Sources

The pipeline is designed to be dataset-agnostic. To incorporate a new disturbance data source:

1.  **Create a Preprocessing Module:**
    *   Add a new Python script in the `src/preprocessing/` directory (e.g., `src/preprocessing/new_dataset.py`).
    *   This script must contain a main processing function (e.g., `process_new_dataset(...)`) that takes raw data paths and other necessary parameters.
    *   The function must output a GeoParquet file to the `outputs/preprocessing/` directory.
    *   **Crucially, this GeoParquet file MUST contain the standard columns:** `geometry`, `start_date`, `end_date`, `mid_date`, `class`, `dataset`, and `year`, with data types and conventions matching the existing datasets (see "Core Workflow" above).
    *   The `class` column must contain values from the `FINAL_TARGET_CLASSES` list in `src/config/constants.py`.

2.  **Update Class Mappings:**
    *   In `src/config/constants.py`, add a new entry to the `RAW_TO_FINAL_TARGET_MAPPINGS` dictionary for your new dataset. This entry will map the raw class labels from your new dataset to the standardized `FINAL_TARGET_CLASSES`.
    *   Example:
        ```python
        # In src/config/constants.py
        RAW_TO_FINAL_TARGET_MAPPINGS = {
            # ... existing mappings ...
            'new_dataset_name': {
                'raw_label_A': 'Fire',
                'raw_label_B': 'Storm',
                '_default_': 'Unknown' # Optional: a default mapping
            }
        }
        ```

3.  **Integrate into Main Preprocessing Script:**
    *   Modify `src/inference/preprocess_excerpts.py`:
        *   Import your new preprocessing function (e.g., `from src.preprocessing import new_dataset`).
        *   Add a new section in the `main()` function to call `new_dataset.process_new_dataset(...)`, providing paths to its raw data and the desired output path in `outputs/preprocessing/`.
        *   Ensure it handles year filtering if applicable.
        *   Add logging and QC calls similar to other datasets.

4.  **Run the Pipeline:**
    *   Once the above steps are done, `src/inference/perform_attribution.py` should automatically detect and load the preprocessed file for your new dataset from `outputs/preprocessing/` (as it typically loads all `.parquet` files from that directory). Its data will then be included in the graph building and attribution process.

## Prerequisites and Installation

*   Python (version 3.9+ recommended).
*   Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    pip install -r requirements.txt
    ```

## Running the Pipeline

1.  **Prepare Raw Data:**
    *   Place your raw data excerpts or full datasets in accessible locations.
    *   Update paths in `src/inference/preprocess_excerpts.py` if they differ from the `excerpts/raw/` structure.

2.  **Run Preprocessing:**
    *   This script processes all datasets and applies date filtering.
    ```bash
    python src/inference/preprocess_excerpts.py --start-year YYYY --end-year YYYY
    ```
    *   Replace `YYYY` with your desired start and end years (e.g., `2016` and `2021`).
    *   Processed files will be saved in `outputs/preprocessing/`.
    *   Logs, including class distributions for each dataset, will be printed to the console.

3.  **Run Attribution:**
    *   This script takes the preprocessed files and performs the attribution.
    ```bash
    python src/inference/perform_attribution.py
    ```
    *   Results (attributed events, cluster data, diagnostic plots) will typically be saved in `outputs/attribution/` and `outputs/qc/`.

## Directory Structure

```
Disturbance-Attribution-Dataset-Joining/
├── excerpts/                   # Raw data excerpts for development
│   └── raw/
├── outputs/                    # Generated files from pipeline runs
│   ├── preprocessing/          # Preprocessed parquet files per dataset
│   ├── attribution/            # Attribution results
│   └── qc/                     # Quality control reports and plots
├── src/                        # Main source code
│   ├── attribution/            # Core attribution logic (pipeline.py)
│   ├── config/                 # Configuration files (constants.py)
│   ├── inference/              # Top-level scripts to run pipeline stages
│   ├── preprocessing/          # Modules for preprocessing individual datasets
│   └── qc/                     # Quality control and diagnostic scripts
├── tests/                      # Unit tests
├── .gitignore
├── LICENSE
├── progress.md                 # Log of development progress and decisions
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Development Log

Please refer to `progress.md` for a detailed log of the development steps, decisions made, and the evolution of the repository.

## Testing

To run unit tests:
```bash
python -m unittest discover tests
```

Good luck, Jorunn! Feel free to reach out if you have questions as you dive in.


