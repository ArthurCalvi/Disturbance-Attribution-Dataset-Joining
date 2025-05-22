# Context 

This repository is used for my PhD research about Forest Disturbances. 

It has two main objectives : 
- Preprocess datasets
- Join datasets by using Louvain Communities and HDBSCAN in order to have better information on disturbance events in France

It has been coded while ago. It's very messy. I want to refactor eveything to have a more clean and lean code that respects PEP8 and best practices. 

# Repository structure 

Old codes can be found in the following folders : 
- annotation/
- join_datasets/
- process_datasets/
- results/
- sampling/
- visualisation/

In join_datasets, experiences.ipynb defines the latest experiments conducted to build this pipeline : Once de the datasets are preprocessed we can join them using Louvain Communities and then apply HDBSCAN on those communities to get better information on each disturbance event. 

# Aim 

New code, simplified, relying on PEP8 and OOP. 

**Structure :** 
excerpts/ 
src/
- __init__.py 
- preprocessing/
- join/
- utils.py
- constants.py
tests/
results/

  
## Pipeline Overview

```mermaid
flowchart TD
    A[Raw disturbance datasets] --> B[Preprocessing notebooks]
    B --> C[Processed .parquet files]
    C --> D[constants.py loading_dict]
    D --> E[Attribution class]
    E --> F[Spatial/Temporal Join]
    F --> G[Louvain communities]
    G --> H[HDBSCAN/DBSCAN clustering]
    H --> I[Clusters & polygons]
    I --> J[results/ visualisation]
```

The notebooks under `process-datasets/` generate simplified Parquet files. They
are loaded according to paths defined in `join-datasets/constants.py`. The
`Attribution` (or `Attribution2`) class performs a spatial join with a tree-cover
loss reference grid, builds similarity matrices and clusters events. Community
detection relies on Louvain; final clusters are produced with DBSCAN or HDBSCAN.

## Latest Experiment Notes

The notebook `join-datasets/experiences.ipynb` contains the most up‑to‑date
experiments. During graph construction the algorithm adaptively increases
thresholds when the graph is not connected:

```text
graph not connected, new thresholds : 1200m, 720d
graph not connected, new thresholds : 2400m, 1440d
graph not connected, new thresholds : 4800m, 2880d
graph not connected, new thresholds : 9600m, 5760d
```

Islands built from those graphs show a very high conversion rate with median
cluster size growing with the aggregation resolution:

```text
conversion rate : 99.84%
median : 5.0, std : 41.61933939084472
conversion rate : 99.84%
median : 7.0, std : 23.250562281483898
conversion rate : 99.84%
median : 9.0, std : 17.4255867488069
conversion rate : 99.84%
median : 9.0, std : 14.45200604486936
conversion rate : 99.84%
median : 10.0, std : 12.699904343020236
```

## Refactoring Guidelines

- Keep only the `_v2` implementations in `join-datasets/utils.py`.
  These include `build_temporal_matrix_v2`, `compute_tree_coherence_v2`,
  `compute_class_similarity_v2`, `compute_spatial_distance_v2`,
  `build_custom_matrix_v2`, `get_matrices_v2`, `build_similarity_v2`,
  `get_temporal_period_v2`, `get_spatial_polygon_v2` and `get_cluster_v2`.
-  The `Attribution2` class should be rewritten and simplified along the method.
- Target structure is `src/` with OOP modules and PEP8 compliant code.

## Datasets

Only the following data will be used : 
- cdi
- hm
- Senf&Seidl
- firepolygons

Other data sources are not relevant or have too much uncertainties. 

