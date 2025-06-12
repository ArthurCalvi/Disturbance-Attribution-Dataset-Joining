# Attribution Pipeline Methodology – Version 3 (24 May 2024)

A streamlined graph-based workflow for attributing causes to **Senf & Seidl** disturbance polygons using four auxiliary datasets (HM, Fire polygons, CDI, FORMS). This document details the scientific methodology and parameters. For a user-centric guide on running the pipeline, see the main `README.md`.

---

## 1. Pipeline Outline

1.  **Neighbour Search** – Find candidate event pairs within a maximum distance and time frame; weight their proximity using an exponential decay function.
2.  **Graph Construction** – Build a graph where nodes are disturbance events and edges are weighted by spatio-temporal proximity, dataset reliability, and an intra-dataset penalty.
3.  **Louvain Community Detection** – Perform a coarse-grained clustering of events into "disturbance complexes" using the Louvain algorithm.
4.  **HDBSCAN Refinement** – Further refine each community in a 4-dimensional space (x, y, scaled time, cause) to identify dense, meaningful clusters.
5.  **Reliability-Weighted Voting** – Attribute a final cause to each Senf & Seidl polygon based on a reliability-weighted vote from all members of its final cluster.

---

## 2. Proximity Weights & Candidate Search

### Spatial Search
*   **Phase 1 (Filter)**: For each event, potential neighbors are found using a spatial index (R-tree) to identify all other events within a bounding box defined by `max_spatial_dist_m`.
*   **Phase 2 (Exact Distance)**: For candidate pairs, the minimum boundary-to-boundary distance is calculated.
    `d_s = geom_a.distance(geom_b)` (after reprojection to a projected CRS like EPSG:2154).

The spatial proximity weight is then calculated as:  
$w_s = \exp(-d_s / L_d)$  
where the half-life **L\_d** (`spatial_half_life`) controls the decay.

### Temporal Search
Events are filtered to keep pairs within `max_temporal_dist_days`. The temporal distance $d_t$ is the number of days between the mid-dates of two events.

The temporal proximity weight is:  
$w_t = \exp(-d_t / L_t)$  
where **L\_t** (`temporal_half_life`) is the temporal half-life.

The combined proximity weight is $w_{prox} = w_s \cdot w_t$.

---

## 3. Graph Construction

*   **Node**: Every disturbance feature from all datasets.
*   **Edge Rule**: Connect any two candidate events with an edge.
*   **Edge Weight**:
    $$
    w = w_{prox} \cdot \underbrace{\tfrac{r_X+r_Y}{2}}_{\text{mean reliability}} \cdot \lambda_{intra}^{[X=Y]}
    $$
    where:
    *   $r_X$ = Reliability scalar for the dataset of event X.
    *   $[X=Y]$ = 1 if both events are from the same dataset, otherwise 0.
    *   **λ\_intra** = A penalty factor applied to within-dataset links to discourage the formation of large, single-source clusters.

### Reliability Priors (`reliability`)

| Dataset (`dataset` key) | r      |
| ----------------------- | ------ |
| `firepolygons`          | **1.00** |
| `hm`                    | 0.90   |
| `forms`                 | 0.80   |
| `senfseidl`             | 0.70   |
| `cdi`                   | 0.50   |

---

## 4. Louvain Community Detection

We run community detection using the Louvain algorithm, controlled by a **resolution γ** (`louvain_resolution`). Tuning this parameter helps control the size and number of communities. Higher values lead to more, smaller communities. Each event is assigned a `community_id`.

---

## 5. Refinement via HDBSCAN

### 5.1. Feature Space
For events within a single Louvain community, we construct a 4-dimensional feature vector for clustering:
```
X = [ x,  y,  t_days / α_t,  δ_c · α_c ]
```
*   `x, y`: Coordinates in a projected CRS (e.g., EPSG:2154 metres).
*   `t_days`: Event mid-date relative to the community's median date (in days). **α\_t** (`alpha_t`) scales time to be comparable to spatial distance.
*   `δ_c`: A binary flag, 0 if the events' broad disturbance classes match, 1 otherwise.
*   **α\_c** (`alpha_c`): A "cause penalty" that converts a class mismatch into a spatial-equivalent distance penalty. This encourages causally pure clusters unless spatio-temporal evidence is overwhelmingly strong.

### 5.2. HDBSCAN Parameters
The clustering is controlled by `min_cluster_size` and `min_samples`. These are calculated dynamically for each community based on absolute and relative parameters to handle communities of varying sizes robustly.
*   `min_cluster_size = max(hdbscan_min_cluster_size_abs, hdbscan_min_cluster_size_rel * N_community)`
*   `min_samples = max(hdbscan_min_samples_abs, hdbscan_min_samples_rel * N_community)`

Events not assigned to a cluster are labeled as noise (-1).

---

## 6. Cause Attribution

For each Senf & Seidl polygon:
1.  Identify its final group (either its HDBSCAN cluster or, if it's noise, its Louvain community).
2.  Gather all members of that group.
3.  Each member "votes" for its disturbance class(es) with a weight equal to its dataset's reliability (`r_dataset`).
4.  The Senf & Seidl polygon's original class also contributes to the vote, but its weight is modulated by `senf_self_vote_factor`.
5.  Sum the weighted votes for each final disturbance class and normalize them to produce a final probability vector.

---

## 7. Parameter Cheatsheet

This table summarizes the key hyperparameters, their corresponding variable names in the code (`AttributionParams`), and their default values.

| Symbol / Concept                | `AttributionParams` Variable         | Default Value | Description                                                    |
| ------------------------------- | ------------------------------------ | ------------- | -------------------------------------------------------------- |
| **Search & Edges**              |                                      |               |                                                                |
| Max Spatial Distance            | `max_spatial_dist_m`                 | 1500.0 m      | Max distance to search for neighbors.                          |
| Max Temporal Distance           | `max_temporal_dist_days`             | 720.0 days    | Max time difference to search for neighbors (≈2 years).        |
| Spatial Half-Life (L\_d)        | `spatial_half_life`                  | 300.0 m       | Distance at which spatial proximity weight decays to 0.5.      |
| Temporal Half-Life (L\_t)       | `temporal_half_life`                 | 180.0 days    | Time at which temporal proximity weight decays to 0.5.         |
| Intra-Dataset Penalty (λ\_intra) | `lambda_intra`                       | 0.5           | Multiplier for edge weights within the same dataset.           |
| **Louvain**                     |                                      |               |                                                                |
| Resolution (γ)                  | `louvain_resolution`                 | 1.0           | Controls size/number of communities. Higher = more, smaller.   |
| **HDBSCAN**                     |                                      |               |                                                                |
| Temporal Scaling (α\_t)         | `alpha_t`                            | 10000.0       | Scales time to be comparable to spatial units.                 |
| Cause Penalty (α\_c)            | `alpha_c`                            | 500.0 m       | Spatial-equivalent penalty for mismatching disturbance causes. |
| Min Cluster Size (abs)          | `hdbscan_min_cluster_size_abs`       | 6             | Absolute minimum number of events in a cluster.                |
| Min Cluster Size (rel)          | `hdbscan_min_cluster_size_rel`       | 0.05          | Relative minimum (fraction of community size).                 |
| Min Samples (abs)               | `hdbscan_min_samples_abs`            | 2             | Absolute minimum samples for a point to be a core point.       |
| Min Samples (rel)               | `hdbscan_min_samples_rel`            | 0.02          | Relative minimum samples (fraction of community size).         |
| **Voting**                      |                                      |               |                                                                |
| S&S Self-Vote Weight            | `senf_self_vote_factor`              | 0.3           | Weight factor for a Senf & Seidl polygon's own class vote.     |

---

*Last updated: 2024-05-24 – Aligned parameters with the `src/attribution/pipeline.py` implementation and clarified the overall workflow.*
