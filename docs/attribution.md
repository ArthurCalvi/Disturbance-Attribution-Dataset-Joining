# Attribution Pipeline Guide

This document explains how processed datasets are merged into disturbance clusters. The approach is based on the methodology described in the `draft_article`.

## Overview
1. Load all processed datasets defined in `join-datasets/constants.py`.
2. Build spatial and temporal weighting functions using dataset and disturbance profiles.
3. Connect nearby events in a graph and detect communities.
4. Cluster events inside each community to identify single disturbance occurrences.
5. Export the attributed clusters as Parquet files for further analysis.

The code implementing these steps resides in `join-datasets/attribution.py` and the supporting notebooks under `join-datasets/`.

## Detailed Steps

### Loading Datasets
`constants.py` lists file paths in `loading_dict` and parameters such as dataset accuracy (`doa`), temporal buffers, and class compositions (`DCLASS_SCORE`). The `Attribution` class reads each dataset into GeoDataFrames and stores metadata for later use.

### Building Weighting Functions
The algorithm uses reliability profiles to weigh spatial and temporal distances. Profiles capture both dataset uncertainties and the typical spread of each disturbance class. They are defined in dictionaries `ddataset_profile` and `ddisturbance_profile` using Gaussian or step functions. These functions are combined to convert raw distances into similarity scores.

### Graph Construction
Events (polygons or points) are joined on a reference grid. For each pair of events occurring within given time and space thresholds, an edge is created in the graph. The edge weight is the average of the two weighting functions for those events. This graph encapsulates how likely two observations refer to the same disturbance.

### Community Detection
The Louvain algorithm is applied to the graph to detect groups of events that are strongly connected. Each community potentially represents a disturbance site where multiple datasets overlap in space and time.

### Clustering Disturbance Events
Inside each community, similarity matrices are computed across four dimensions: spatial distance, temporal distance, tree species match and class similarity. An unsupervised clustering algorithm (e.g. HDBSCAN or spectral clustering) then merges events into final disturbance clusters. Each cluster is assigned a predominant disturbance class and a time range.

### Outputs
Results are written as `disturbances_<year>_g<granularity>_v<version>.parquet` and `clusters_<year>_g<granularity>_v<version>.parquet`. These contain the cluster polygons, their classes and confidence scores.

For examples and visualisations refer to the notebooks in `visualisation/` and to the methodological explanation in `draft_article`.
