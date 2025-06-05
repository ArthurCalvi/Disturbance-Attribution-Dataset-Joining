# Context 

This repository is used for my PhD research about Forest Disturbances. 

It has two main objectives : 
- Preprocess datasets
- Join datasets by using Louvain Communities and HDBSCAN in order to have better information on disturbance events in France

The codebase has been fully refactored for clarity, modularity, and PEP8/OOP best practices. All new code is under `src/`.

# Repository Structure (Current)

- `excerpts/` – Raw data excerpts for development
- `src/`
  - `preprocessing/` – Preprocessing modules for each dataset
  - `attribution/` – Graph-based attribution pipeline (Louvain, HDBSCAN, voting)
  - `inference/` – Orchestrates full workflow (preprocessing, attribution)
  - `qc/` – Quality control and diagnostic scripts (plots, stats)
  - `config/` – Central configuration (constants, class mappings)
- `outputs/`
  - `preprocessing/` – Preprocessed GeoParquet files
  - `attribution/` – Attribution results
  - `qc/` – QC plots and diagnostics
- `tests/` – Unit tests
- `progress.md` – Log of all steps and decisions

# Pipeline Overview

```mermaid
flowchart TD
    A[Raw disturbance datasets] --> B[Preprocessing (src/preprocessing/)]
    B --> C[Processed GeoParquet files (outputs/preprocessing/)]
    C --> D[Attribution (src/attribution/)]
    D --> E[Attributed events & clusters (outputs/attribution/)]
    E --> F[QC & Visualisation (src/qc/, outputs/qc/)]
```

- Raw data is excerpted and preprocessed using dedicated modules in `src/preprocessing/`.
- Outputs are standardized GeoParquet files in `outputs/preprocessing/`.
- Attribution is performed using the `Attribution` class in `src/attribution/pipeline.py`, building a spatio-temporal graph, detecting Louvain communities, clustering with HDBSCAN, and attributing causes.
- Results are saved in `outputs/attribution/`.
- QC and visualisation scripts in `src/qc/` generate diagnostics and plots (e.g., community size distributions, preprocessing geometry plots) in `outputs/qc/`.

# Key Features

- Modular, testable, and reproducible pipeline.
- All preprocessing, attribution, and QC logic is in `src/`.
- Centralized class mappings and configuration in `src/config/constants.py`.
- Logging and progress tracking in `progress.md`.
- Unit tests in `tests/`.

# Current Status

- All preprocessing and attribution logic has been refactored and validated.
- The pipeline is fully operational from raw data excerpts to attributed event outputs.
- QC and basic visualisation scripts are in place for both preprocessing and attribution steps.

# Next Mission: Visualisation & Parameter Tuning

- **Enhance visualisation during and after attribution:**
  - Develop richer output plots (e.g., spatial maps of clusters, class distributions, event timelines).
  - Integrate visualisation steps directly into the attribution workflow for immediate feedback.
- **Tune attribution parameters:**
  - Systematically explore Louvain and HDBSCAN parameters using diagnostics.
  - Use QC outputs to guide parameter selection for optimal clustering and attribution.
- **Produce publication-ready output visualisations:**
  - Automate generation of key figures for reports and papers.
  - Ensure all outputs are saved in `outputs/qc/` and are reproducible from the pipeline.

# Documentation

- See `README.md` for a user/developer guide and pipeline details.
- See `progress.md` for a chronological log of all steps, issues, and decisions.

# Datasets Used

- cdi (droughts)
- hm (biotic agents)
- Senf & Seidl (segmentation and classification)
- firepolygons (fires)
- forms (clear cut)

Other data sources are not currently included.

