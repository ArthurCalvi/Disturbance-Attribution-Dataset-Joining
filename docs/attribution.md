# Lean Attribution Pipeline – Version 3 (23 May 2025)

A streamlined graph‑based workflow for attributing causes to **Senf & Seidl** disturbance polygons using four auxiliary datasets (HM, Fire polygons, CDI, FORMS). Emphasis: minimal arbitrary parameters, transparent science, maintainable code.

---

## 1  Pipeline Outline

1. **Neighbour search** – find candidate pairs within 5 km / ±2 y; weight proximity with exponential decay.
2. **Graph construction** – weighted edges between *all* datasets (cross‑ & intra‑source) using dataset reliability.
3. **Louvain communities** – coarse grouping of events into disturbance complexes.
4. **Mandatory HDBSCAN** – refine each community in 4‑D space–time–cause feature space.
5. **Reliability‑weighted voting** – per Senf & Seidl polygon, output a probability vector that sums to 1.

---

## 2  Proximity Weights

### Spatial distance

* **Phase 1 (filter)**: centroid–to‑centroid haversine distance via GeoPandas R‑tree. Keep pairs ≤ 6 km.
* **Phase 2 (exact)**: minimum boundary‑to‑boundary distance (0 m if geometries intersect).
  `d_s = geom_a.distance(geom_b)` after reprojection to EPSG:2154.

Weight  $w_s = \exp(-d_s / L_d)$ ; default half‑life **L\_d = 1000 m** (weight 0.5 at 1.0 km).

### Temporal distance

Treat each event as interval $[t_{start}, t_{end}]$.

* If intervals overlap ⇒ $d_t=0$.
* Else $d_t = \min(|t_{start}-t'_{end}|, |t_{end}-t'_{start}|)$ in **days**.

Weight  $w_t = \exp(-d_t / L_t)$ ; default half‑life **L\_t = 180 d** (≈6 months).

Combined proximity weight  $w_{prox} = w_s · w_t$.

---

## 3  Graph Construction

* **Node** = every disturbance feature from the five datasets.
* **Edge rule**: connect any two events with $w_{prox}>0$.
  Edge weight:

  $$
  w = w_{prox} · \underbrace{\tfrac{r_X+r_Y}{2}}_{\text{mean reliability}} · \lambda_{intra}^{[X=Y]}
  $$

  where

  * $r_X$ = reliability scalar per dataset
  * $[X=Y]$ = 1 if same dataset, else 0
  * **λ\_intra = 0.5** (down‑weights within‑dataset links to prevent giant single‑source blobs).

Reliability priors:

| Dataset                       | r        |
| ----------------------------- | -------- |
| Fire polygons                 | **1.00** |
| HM                            | 0.90     |
| FORMS                         | 0.80     |
| Senf & Seidl (self‑vote only) | 0.70     |
| CDI                           | 0.50     |

Implementation: build edges lazily by iterating over the candidate pairs list; store graph in `networkx.Graph` with weights as edge attr.

---

## 4  Louvain Community Detection

Run `community-louvain` with a **resolution γ** (default 1.0). Tune γ until these heuristics hold:

* median community diameter < 3 km,
* 90‑percentile community size < 200 nodes,
* modularity > 0.35.
  Each node gets `community_id` attribute.

---

## 5  Refinement via HDBSCAN (Mandatory)

### 5.1  Feature vector

For events in one Louvain community:

```
X = [ x , y ,  t_days / α_t ,  δ_c · α_c ]
```

* `x, y` – EPSG:2154 metres.
* `t_days` – event mid‑date relative to community median (days).  **α\_t = 10000** converts 1 year ≈ 0.1 km in the distance metric.
* `δ_c` = 0 if broad class matches, 1 otherwise (fire vs non‑fire, biotic vs non‑biotic, etc.).
* **α\_c = 500 m** converts cause mismatch into a *spatial equivalent penalty* – two events with different causes must be ≈500 m closer in the other dimensions to offset this extra distance.

### 5.2  HDBSCAN parameters

* `min_cluster_size = max(6, 0.05 · N_comm)`
* `min_samples = max(2, 0.02 · N_comm)`
  Distance metric = Euclidean on X.
  Outputs `hdb_id` per event; noise labelled −1.

### 5.3  Why include cause penalty?

* **Negligible cost**: one extra numeric column.
* **Purity boost**: discourages inadvertent fusion of, e.g., fire polygons with insect HM points when spatial evidence is borderline.
* **Soft constraint**: α\_c is moderate; if a fire and a salvage‑logging FORMS clear‑cut truly overlap (same polygon), spatial term 0 overrides penalty and they cluster.

---

## 6  Cause Attribution

For each Senf & Seidl polygon:

1. Gather all members of its `hdb_id` (or `community_id` if HDBSCAN labelled it noise).
2. Each member casts a vote for its **broad class** with weight `r_dataset`.
3. Add Senf & Seidl’s own label with weight `0.3 · r_S`.
4. Sum weights per class and normalise to 1.
   Return

```python
{senf_id: {"fire": p_fire, "biotic": p_bio, ...}}
```

---

## 7  Parameter Cheatsheet & Tuning

| Symbol      | Default | Tune by …                                                          |
| ----------- | ------- | ------------------------------------------------------------------ |
| L\_d        | 1000 m  | maximise % S\&S with ≥1 match but keep median comm. diameter <3 km |
| L\_t        | 180 d   | examine Δt histogram; <5 % links over 2 y                          |
| λ\_intra    | 0.5     | raise if same‑dataset duplicates still split; lower if over‑merge  |
| γ (Louvain) | 1.0     | satisfy diameter/size/modularity heuristics                        |
| α\_t        | 10000   | adjust so 1 y ≈ 0.05–0.2 km equivalence                            |
| α\_c        | 500 m   | increase for stricter cause homogeneity                            |

---

## 8  Evaluation Metrics

* **Cause‑purity entropy** within each HDBSCAN cluster ↓
* **Attribution coverage**: % S\&S polygons with external evidence ↑
* **Known‑event check**: wildfire list ⇒ p\_fire ≥ 0.9

---

## 9  Code Skeleton

```python
class Attribution:
    def __init__(self, gdf_dict, params=DEFAULTS):
        self.data = gdf_dict
        self.p = params
    def build_edges(self):
        # 1. neighbour search + proximity weights
        # 2. edge weight = prox * reliability * λ_intra
    def build_graph(self):
        self.G = nx.Graph()
        self.G.add_weighted_edges_from(self.build_edges())
    def louvain(self):
        self.comms = louvain_communities(self.G, resolution=self.p.gamma)
    def hdb(self):
        for comm in self.comms:
            self._hdbscan_cluster(comm)
    def attribute(self):
        return self._vote_per_senf()
```

Functions are pure; every step dumps interim GeoPackages (or .parquet for speed) for QC.

---

*Last updated: 2025‑05‑23 18:40 UTC – incorporates clarity on cause‑penalty and x/y usage.*
