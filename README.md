# Disturbance-Attribution-Dataset-Joining


## Required Dataset Columns

During preprocessing, ensure each output dataset provides at least the following columns so the joining algorithms work correctly:

- `geometry` – polygon or point geometry in EPSG:2154
- `start_date` and `end_date` – disturbance time range
- `class` – disturbance class
- `dataset` – dataset name (added automatically by the loader)
- `year` – fallback year used when detailed dates are missing


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

This repository contains a Python-based pipeline for attributing forest disturbances by integrating and analyzing data from multiple sources. It aims to provide a clearer understanding of disturbance events by combining the strengths of various datasets, with a focus on attributing causes to events in a reference dataset (e.g., Senf & Seidl).

## Core Workflow

The pipeline operates in two main phases, orchestrated by scripts in `src/inference/`:

### 1. Preprocessing (`preprocess_excerpts.py`)
This phase takes raw disturbance data from various sources (e.g., Senf & Seidl rasters, fire polygons, health monitoring reports) and standardizes them.

- **Harmonization of Classes**: A key step is mapping raw disturbance classes from each source to a final, unified set of classes. This is configured in `src/config/constants.py`.
- **Non-Injective Mapping**: The pipeline now supports **non-injective (many-to-one) mappings**. A single raw class that represents multiple potential causes (e.g., 'Other' or 'Storm,Biotic') can be mapped to a list of final classes (e.g., `['Anthropogenic', 'Unknown', 'Drought']`). This preserves ambiguity for the attribution phase.
- **Output**: This phase produces a set of standardized GeoParquet files in `outputs/preprocessing/`, ready for attribution.

### 2. Attribution (`perform_attribution.py`)
This phase loads the preprocessed data and performs the core analysis using the `Attribution` class from `src/attribution/pipeline.py`.

- **Graph-Based Analysis**: It builds a spatio-temporal graph of all disturbance events to find relationships.
- **Community Detection**: The Louvain algorithm is used to detect communities of related events.
- **Clustering**: Within each community, HDBSCAN is applied to find dense spatio-temporal-causal clusters.
- **Attribution Logic**: The final cause of a reference event (e.g., from Senf & Seidl) is determined by a voting mechanism based on the classes of other events in its cluster and community. Events with ambiguous classes that are isolated (i.e., not clustered with events from other datasets) are intelligently defaulted to 'Unknown'.
- **Output**: The final output is a single GeoDataFrame (`final_attributed_disturbances.parquet`) in `outputs/attribution/` containing all events with newly assigned attribution probabilities and final classes.

## Automated Quality Control & Visualization
Running the attribution pipeline automatically generates a suite of professional, GIS-style plots in the `outputs/qc_attribution/` directory, providing immediate insight into the results.

- **Attributed Class & Confidence Maps**: Spatial plots showing the final attributed class and confidence score for each event.
- **Class Distribution Charts**: Pie and bar charts showing the overall distribution of attributed classes.
- **NEW: Community Envelope Plot**: A map of community envelopes (convex hulls), colored by the number of events within each community. This provides a clear view of the spatial clustering performed by the Louvain algorithm.

## Running the Pipeline

### 1. Prerequisites
- Python (3.9+ recommended)
- Setup a virtual environment and install dependencies:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Linux/macOS
  # venv\Scripts\activate    # On Windows
  pip install -r requirements.txt
  ```

### 2. Prepare Raw Data
- Place raw data in the `excerpts/raw/` directory or update paths in `src/inference/preprocess_excerpts.py`.

### 3. Run Preprocessing
This script processes all datasets. The year filter is mandatory.
```bash
python src/inference/preprocess_excerpts.py --start-year YYYY --end-year YYYY
```
- Replace `YYYY` with your desired start and end years (e.g., `2016` and `2021`).
- Outputs are saved in `outputs/preprocessing/`.

### 4. Run Attribution
This script runs the main analysis and generates the final outputs and visualizations.
```bash
python src/inference/perform_attribution.py
```
- **Flags for controlling caching:**
  - `--clear-cache`: Deletes the `outputs/temp_attribution` cache before running.
  - `--recompute-graph`: Forces the graph to be rebuilt, ignoring any cached version.
  - `--recompute-communities`: Forces community detection to be re-run.

## Key Configuration & Tuning

- **Class Mappings (`src/config/constants.py`):** The `RAW_TO_FINAL_TARGET_MAPPINGS` dictionary is the most critical place for configuration. This is where you define how raw classes from input datasets are mapped to the final, standardized classes, including the non-injective list-based mappings for ambiguous classes.
- **Community Detection (`src/attribution/pipeline.py`):** The `louvain_resolution` parameter in `AttributionParams` controls the size and number of communities. Use the community size diagnostic plot (`outputs/qc/community_size_distribution.png`) to help tune this.
- **HDBSCAN Clustering (`src/attribution/pipeline.py`):** The `_hdbscan_cluster` method contains parameters like `min_cluster_size` that control the density of final clusters.

## Extending the Pipeline (Guide for New Datasets)

To add a new data source:
1.  **Create a Preprocessing Module:** Add a `your_dataset.py` script in `src/preprocessing/`. This script must have a function that outputs a GeoParquet file to `outputs/preprocessing/` with the standard columns (`geometry`, `start_date`, `end_date`, `class`, `dataset`, etc.).
2.  **Update Class Mappings:** In `src/config/constants.py`, add an entry to `RAW_TO_FINAL_TARGET_MAPPINGS` for your new dataset to map its raw classes to the pipeline's final classes.
3.  **Integrate into Main Script:** In `src/inference/preprocess_excerpts.py`, import and call your new preprocessing function.

The attribution pipeline will automatically detect and use the new preprocessed file.

## Directory Structure

```
Disturbance-Attribution-Dataset-Joining/
├── excerpts/                   # Raw data excerpts for development
├── outputs/                    # Generated files from pipeline runs
│   ├── preprocessing/          # Standardized parquet files per dataset
│   ├── attribution/            # Final attribution results
│   └── qc_attribution/         # Automatically generated plots and reports
├── src/                        # Main source code
│   ├── attribution/            # Core attribution logic (pipeline.py)
│   ├── config/                 # Configuration files (constants.py)
│   ├── inference/              # Top-level scripts to run pipeline stages
│   ├── preprocessing/          # Modules for preprocessing individual datasets
│   └── qc/                     # QC scripts and helpers
├── tests/                      # Unit tests
├── progress.md                 # Detailed log of development progress
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Testing
To run unit tests:
```bash
python -m unittest discover tests
```


