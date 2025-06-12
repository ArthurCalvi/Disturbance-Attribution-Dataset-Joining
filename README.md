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
- **Community Detection**: The Louvain algorithm is used to detect communities of related events. It operates on the graph, where edge weights are a function of spatial proximity (distance between polygons), temporal proximity (time between disturbances), and the reliability of the source datasets. This step groups events into coarse "disturbance complexes."
- **Clustering**: Within each community, HDBSCAN is applied to find dense spatio-temporal-causal clusters. This is a refinement step that operates in a 4-dimensional feature space:
  1.  `x` coordinate (from geometry centroid)
  2.  `y` coordinate (from geometry centroid)
  3.  `scaled time` (event date relative to its community's median date, scaled to be comparable to spatial units)
  4.  `cause code` (a numeric representation of the disturbance class)
- **Attribution Logic**: A custom distance metric is used within HDBSCAN. The distance between two events is the normal Euclidean distance in the 3D `(x, y, scaled_time)` space, but with a large penalty (`--alpha-c`) added if their cause codes differ. This makes it difficult—but not impossible—for events of different types to be clustered together, encouraging causally pure clusters unless the spatio-temporal evidence is overwhelmingly strong. The final cause of a reference event (e.g., from Senf & Seidl) is determined by a voting mechanism based on the classes of other events in its cluster and community. Events with ambiguous classes that are isolated (i.e., not clustered with events from other datasets) are intelligently defaulted to 'Unknown'.
- **Output**: The final output is a single GeoDataFrame (`final_attributed_disturbances.parquet`) in `outputs/attribution/` containing all events with newly assigned attribution probabilities and final classes.

## Automated Quality Control & Visualization
The pipeline includes automated steps to generate plots for quality control and immediate visual feedback. These plots help monitor the process and optimize parameters.

### 1. Preprocessing QC (from `preprocess_excerpts.py`)
After each dataset is preprocessed, a quality control report is generated in a corresponding subfolder within `outputs/qc_preprocessing/`. These reports are useful for verifying that each dataset was read and processed correctly before moving to the attribution phase. They typically include:
- A spatial plot showing the distribution of the processed geometries.
- A summary of key attributes, such as the distribution of disturbance classes.

### 2. Attribution QC (from `perform_attribution.py`)
Running the attribution pipeline generates a suite of more advanced plots, providing insight into the intermediate and final results.

**Community Diagnostics (in `outputs/qc/`)**
- **Community Size Distribution**: A histogram showing the number of events in each detected Louvain community. This plot is essential for tuning the `--louvain-resolution` parameter. An ideal distribution avoids having a single giant community, favoring many small to medium-sized ones instead.

**Final Attribution Results (in `outputs/qc_attribution/`)**
- **Attributed Class & Confidence Maps**: Spatial plots showing the final attributed class and confidence score for each event.
- **Class Distribution Charts**: Pie and bar charts showing the overall distribution of attributed classes.
- **Community Envelope Plot**: A map of community envelopes (convex hulls), colored by the number of events within each community. This provides a clear view of the spatial clustering performed by the Louvain algorithm.

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

Alternatively, you can download the raw data excerpts for the 'les landes' bounding box from Google Drive using the following command:
```bash
python src/utils/download.py --url "https://drive.google.com/file/d/1LMMDZqEWz5ee8nYcjRH3K2Uch7oT0jmb/view?usp=share_link" --output "excerpts/raw/""
```

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
The attribution pipeline uses an intelligent caching system to speed up subsequent runs. Intermediate results (like the computed graph and community data) are stored in `outputs/temp_attribution/`.

- **Automatic Cache Validation**: The script saves the parameters used for a successful run in `outputs/temp_attribution/params.json`. Before a new run, it compares the current parameters with the saved ones. If any parameter has changed, **the cache is automatically cleared** to prevent using stale data. This ensures results are always consistent with the parameters.
- **Manual Cache Control**: You can override the automatic behavior with these flags:
  - `--clear-cache`: Deletes the `outputs/temp_attribution` cache before running.
  - `--recompute-graph`: Forces the graph to be rebuilt, ignoring any cached version.
  - `--recompute-communities`: Forces community detection to be re-run.

## Key Configuration & Tuning

The pipeline's behavior can be fine-tuned through two primary mechanisms: editing the class mappings and providing command-line arguments to `perform_attribution.py`.

- **Class Mappings (`src/config/constants.py`):** The `RAW_TO_FINAL_TARGET_MAPPINGS` dictionary remains the most critical place for semantic configuration. This is where you define how raw classes from input datasets are mapped to the final, standardized classes.

- **Pipeline Hyperparameters (Command-Line)**: Instead of editing the source code, you can now tune all key algorithm parameters directly from the command line when running `perform_attribution.py`. The script will show the default values if you run it with `--help`. If a parameter is not specified, its default value is used.

  Here are the most important parameters you can tune:

  - **Graph Construction:**
    - `--spatial-half-life`: Spatial decay half-life in meters (default: 300.0).
    - `--temporal-half-life`: Temporal decay half-life in days (default: 180.0).
    - `--max-spatial-dist-m`: Maximum spatial distance to consider pairing events (default: 1500.0).
    - `--max-temporal-dist-days`: Maximum temporal distance to consider pairing events (default: 720.0).
    - `--lambda-intra`: Down-weighting factor for links within the same dataset (default: 0.5).
      > **Note on Multi-Year Disturbances and `--lambda-intra`**:
      > A key challenge is attributing separate causes when a single polygon (e.g., from Senf & Seidl) is disturbed in multiple years (e.g., a drought in 2018 followed by a biotic attack in 2020).
      > Setting `--lambda-intra 0` is the first step, as it prevents these recurring events from being directly linked in the graph. However, this does **not** guarantee independent attribution.
      > **Caveat**: The two events can still be indirectly connected if they both share a common neighbor from another dataset. This will likely place them in the same community and even the same final cluster, causing their attributions to be influenced by each other.
      > Achieving truly independent attribution for each year would require modifications to the core pipeline logic to handle such cases specifically. This is a known limitation and an area for future improvement.

  - **Community Detection:**
    - `--louvain-resolution`: Resolution for Louvain community detection. Higher values lead to more, smaller communities (default: 1.0).

  - **HDBSCAN Clustering:**
    - `--alpha-c`: The "cause penalty" in meters. A higher value makes it harder to cluster events of different types (default: 500.0).
    - `--hdbscan-min-cluster-size-abs`: Absolute minimum number of events in a valid cluster (default: 6).
    - `--hdbscan-min-samples-abs`: Absolute minimum number of samples for a point to be a core point (default: 2).

  - **Attribution Logic:**
    - `--senf-self-vote-factor`: The weight factor for a Senf&Seidl polygon's own class when voting (default: 0.3).

  **Example of a tuned run:**
  ```bash
  python src/inference/perform_attribution.py --spatial-half-life 400 --louvain-resolution 1.1
  ```

## Extending the Pipeline: Adding a New Dataset

The pipeline is designed to be modular, allowing for the straightforward addition of new disturbance datasets. Each dataset has its own preprocessing logic defined in a dedicated module within `src/preprocessing/`, which is then orchestrated by the main `src/inference/preprocess_excerpts.py` script.

Here is a step-by-step guide to integrate a new dataset:

### Step 1: Add Raw Data
1.  Create a new folder for your dataset inside `excerpts/raw/`. The name of the folder should be a simple identifier for your dataset (e.g., `excerpts/raw/my_new_dataset/`).
2.  Place all the raw data files for your dataset into this new folder.

### Step 2: Create a Preprocessing Module
1.  Create a new Python file in `src/preprocessing/` (e.g., `src/preprocessing/my_new_dataset.py`).
2.  In this file, define a primary function that will contain all the logic for processing the raw data (e.g., `process_my_new_dataset()`).
3.  This function should accept arguments for the input file path(s), the output file path, and any filters, such as `start_year` and `end_year`.

### Step 3: Implement the Preprocessing Logic
Inside your `process_my_new_dataset` function, you will load, transform, and save the data.

1.  **Load Data**: Read your raw data file(s) (e.g., rasters, shapefiles, CSVs).
2.  **Transform and Harmonize**: Perform the necessary cleaning, reprojection, and data transformation.
3.  **Standardize Output**: The core goal is to produce a `GeoDataFrame` that adheres to the project's standard schema. **Crucially, the final GeoDataFrame must include the columns listed in the "Required Dataset Columns" section of this README.** The `geometry` column must be in `EPSG:2154`.
4.  **Map Classes**: In `src/config/constants.py`, add a new entry to the `RAW_TO_FINAL_TARGET_MAPPINGS` dictionary. This entry will map the raw disturbance classes from your new dataset to the pipeline's standardized classes. Your preprocessing function should use this mapping to populate the `class` column.
5.  **Save Output**: Save the final, standardized `GeoDataFrame` to the provided output path as a GeoParquet file.

### Step 4: Integrate into the Main Preprocessing Script
The final step is to call your new module from the main orchestration script, `src/inference/preprocess_excerpts.py`.

1.  **Import Your Module**: At the top of `src/inference/preprocess_excerpts.py`, add an import statement for your new module:
    ```python
    from src.preprocessing import my_new_dataset
    ```
2.  **Add a Processing Block**: In the `main` function of the script, add a new block to handle your dataset. This block will find the raw data, define the output path, and call your processing function. It should follow the pattern of the existing datasets:

    ```python
    # --- My New Dataset ---
    logging.info("--- Starting My New Dataset preprocessing ---")
    try:
        # 1. Define input path from excerpts/raw/
        my_new_dataset_dir = EXCERPTS_RAW_DIR / "my_new_dataset"
        if my_new_dataset_dir.is_dir():
            # 2. Define the standardized output path
            output_path = OUTPUT_PREPROCESSING_DIR / "my_new_dataset_processed.parquet"
            
            # Get input files (example for .tif files)
            input_files = [str(p) for p in my_new_dataset_dir.glob('*.tif')]

            # 3. Call your processing function
            processed_gdf = my_new_dataset.process_my_new_dataset(
                input_files=input_files,
                output_file=str(output_path),
                start_year_filter=start_year_arg,
                end_year_filter=end_year_arg
            )
            logging.info(f"My New Dataset preprocessing finished. Generated {len(processed_gdf)} records.")
            
            # 4. (Optional but recommended) Add QC step
            if not processed_gdf.empty:
                generate_qc_report(str(output_path), "my_new_dataset")

        else:
            logging.warning(f"Skipping My New Dataset, input directory not found: {my_new_dataset_dir}")

    except Exception as e:
        logging.error(f"Error during My New Dataset preprocessing: {e}", exc_info=True)
    logging.info("--- Finished My New Dataset preprocessing ---")
    ```

Once these steps are complete, running `preprocess_excerpts.py` will automatically process your new dataset, and the attribution pipeline will include its data in the analysis.

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



## Required Dataset Columns

During preprocessing, ensure each output dataset provides at least the following columns so the joining algorithms work correctly:

- `geometry` – polygon or point geometry in EPSG:2154
- `start_date` and `end_date` – disturbance time range
- `class` – disturbance class
- `dataset` – dataset name (added automatically by the loader)
- `year` – fallback year used when detailed dates are missing


## Documentation
The preprocessing scripts are described in [docs/preprocessing.md](docs/preprocessing.md).
The attribution workflow is explained in [docs/attribution.md](docs/attribution.md).

## Testing
To run unit tests:
```bash
python -m unittest discover tests
```