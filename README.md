# Disturbance-Attribution-Dataset-Joining

Disturbance Attribution by Dataset Temporo-Spatial Joining. This project joins multiple disturbance detection sources with a common reference so that each event is attributed to a disturbance type.

## Project Objectives
- Preprocess several raw disturbance datasets.
- Join datasets with a reference temporal/spatial grid.
- Attribute disturbances to produce clusters and summary outputs.

## High Level Workflow
1. **Preprocessing**: Run the notebooks in `process-datasets/` to convert raw files to simplified `.parquet` datasets. The notebooks expect the final files to be saved under `data/processed_datasets/`.
2. **Join and Attribution**: Use the notebooks in `join-datasets/` or the `attribution.py` script to perform the temporo–spatial join and cluster attribution. Results are written in a directory you provide when calling the methods.
3. **Visualisation and Analysis**: Additional notebooks in `visualisation/` explore the output clusters and weighting functions.

## Prerequisites and Installation
- Python 3.9 or later.
- Recommended packages include `geopandas`, `dask-geopandas`, `dask[dataframe]`, `pandas`, `numpy`, `matplotlib`, `contextily`, `matplotlib-scalebar`, `joblib`, `tqdm`, and `rasterio`.

Install the dependencies with pip:
```bash
pip install geopandas dask[dataframe] dask-geopandas pandas numpy matplotlib contextily matplotlib-scalebar joblib tqdm rasterio
```

## Running Preprocessing
Open each notebook in `process-datasets/` and run it from top to bottom. The notebooks download or read the raw disturbance data and output simplified `.parquet` files in `data/processed_datasets/`. These file paths correspond to the entries in `join-datasets/constants.py` under `loading_dict`.

## Running the Attribution Script
The `join-datasets/attribution.py` module defines an `Attribution` class. After loading the datasets using the paths in `constants.py`, you can compute clusters for a given year:
```python
from attribution import Attribution
from constants import loading_dict, DCLASS_SCORE, ddataset_profile, ddisturbance_profile

datasets = {name: gpd.read_parquet(path) for name, path in loading_dict.items()}
attr = Attribution(datasets, reference=tcl, doa=doa, dtypes=dtypes,
                   temporal_buffer=temporal_buffer, dsbuffer=dsbuffer,
                   dclass_score=DCLASS_SCORE, ddataset_profile=ddataset_profile,
                   ddisturbance_profile=ddisturbance_profile, start_year=2017)
attr.get_clusters(2017, dcustom_similiraity_function, dir_="../data/results/cluster_2017")
```
Outputs include `disturbances_<year>_g<granularity>_v<version>.parquet` and `clusters_<year>_g<granularity>_v<version>.parquet` saved in the directory you specify.

## Input Data Locations
`join-datasets/constants.py` expects processed datasets at:
```
../data/processed_datasets/simplified_<dataset>_EPSG2154_FR.parquet
```
relative to the `join-datasets` folder. Create the `data/processed_datasets/` directory at the repository root and place the processed `.parquet` files there.

## Disturbance Data Sources
Links for downloading disturbances used in this project:

- **Senf & Seidl maps**
  - Attribution: https://zenodo.org/record/8202241 (use version 1.2 with the Black Beetle & Wind merge)
  - Detection: https://zenodo.org/record/7080016#.Y7QtTS8w30o
- **Patacca et al. record**

