# Guide to Tuning the Attribution Pipeline

This document provides a practical guide for tuning the hyperparameters of the attribution pipeline. The goal is to calibrate the model to produce ecologically meaningful results that align with your understanding of forest disturbances. The process is iterative and combines expert knowledge, visual inspection of outputs, and quantitative metrics.

This guide is intended for the data scientist or ecologist responsible for calibrating the model's performance. It assumes familiarity with the pipeline's methodology as described in `docs/ATTRIBUTION.md`.

---

## General Philosophy

Tuning is a balancing act. There is no single "correct" set of parameters; the optimal configuration depends on the specific characteristics of your input datasets and the ecological processes you are studying. The recommended approach is to work sequentially through the pipeline's stages, making informed adjustments at each step.

**Key Tools for Tuning:**
*   **QC Plots:** The automatically generated plots for community size and final attributions are your primary source of feedback.
*   **Visual Inspection:** Regularly plotting the spatial outputs (community envelopes, cluster members) in a GIS environment is crucial for building intuition and spotting anomalies.
*   **Metrics:** Use the evaluation metrics (cause-purity, coverage) described in `docs/ATTRIBUTION.md` to guide your final adjustments.

---

## 1. Tuning Graph Construction Parameters

These parameters define the fundamental connections between disturbance events. They should be based on physical and ecological reasoning about the spatio-temporal scale of disturbance processes.

### Candidate Search Filters (`max_spatial_dist_m`, `max_temporal_dist_days`)

*   **Concept:** These are hard cutoffs that create a "search radius" around each event to find potential neighbors. Their main purpose is to reduce the computational load by excluding obviously unrelated pairs.
*   **Tuning Strategy:**
    *   These values should be set generously. Think about the largest plausible extent of a "disturbance complex." For example, if a large windstorm is followed by salvage logging and bark beetle outbreaks over two years across a 2km valley, your parameters must be large enough to capture this.
    *   `max_spatial_dist_m` (Default: 1500m): A value between 1,500m and 5,000m is often reasonable.
    *   `max_temporal_dist_days` (Default: 720d): A value between 720 days (2 years) and 1095 days (3 years) is a common starting point.
    *   **Action:** Set these once at the beginning based on ecological reasoning and leave them fixed. They are not intended for fine-tuning.

### Proximity Half-Lives (`spatial_half_life`, `temporal_half_life`)

*   **Concept:** These parameters control the exponential decay of the proximity weight. The `half_life` is the distance (in meters or days) at which the weight between two events drops to 0.5. They are the most important parameters for defining what "close" means.
*   **Tuning Strategy:**
    *   **`spatial_half_life` (L_d, Default: 300m):** This should reflect the characteristic spatial scale of influence. Ask: "At what distance are two events still strongly related?" If two fire polygons 500m apart are very likely part of the same event complex, a half-life of 300-500m might be appropriate.
    *   **`temporal_half_life` (L_t, Default: 180d):** This should reflect the typical timescale of cascading events. For example, if salvage logging typically occurs within 6 months of a storm, a half-life of 180 days makes sense.
    *   **Action:** Analyze the distance/time between events you *know* should be linked. Plot histograms of these distances. Adjust the half-lives so that these known pairs receive a reasonably high proximity score (e.g., >0.25).

### Intra-Dataset Penalty (`lambda_intra`)

*   **Concept:** This factor (Default: 0.5) reduces the weight of edges connecting two events from the *same* dataset. It prevents large, homogeneous polygons from a single source from forming a single, un-influenceable blob.
*   **Tuning Strategy:**
    *   If you observe that large events from a single dataset (e.g., a massive Senf & Seidl drought polygon) are dominating their communities and not properly associating with smaller events from other datasets, **decrease `lambda_intra`** (e.g., to 0.1-0.3).
    *   If events from the same dataset that should clearly be linked are being split apart too easily, you could slightly **increase `lambda_intra`**.
    *   **Action:** Inspect the `dataset` composition of several medium-sized communities. If they are heavily dominated by a single source, `lambda_intra` may be too high.

---

## 2. Tuning Louvain Community Detection (`louvain_resolution`)

*   **Concept:** The resolution parameter (γ, Default: 1.0) controls the granularity of the Louvain algorithm. Higher values result in more, smaller communities.
*   **Tuning Strategy:** This is a direct trade-off between community size and modularity. The goal is to find a "sweet spot."
    *   **Use the QC Plot:** Examine the "Community Size Distribution" histogram generated by the pipeline.
    *   **Avoid Pathologies:**
        *   If the plot shows one giant community containing most of the events, **increase `louvain_resolution`** (e.g., 1.1, 1.2, 1.5).
        *   If the plot shows thousands of tiny communities (e.g., most have <5 members), **decrease `louvain_resolution`** (e.g., 0.9, 0.8).
    *   **Ecological Check:** A community should represent a plausible "disturbance complex." Plot the convex hulls of a dozen sample communities on a map. Do they make sense spatially and temporally? Are they capturing a single storm, fire, or outbreak?
    *   **Action:** Start with the default and increment/decrement by 0.1 until the size distribution looks reasonable (e.g., a long tail with a median size of 10-50 events).

---

## 3. Tuning HDBSCAN Clustering

These parameters fine-tune the clusters *within* each Louvain community.

### Causal-Spatial Scaling (`alpha_c`, `alpha_t`)

*   **Concept:** These parameters balance the dimensions of the feature space (x, y, time, cause).
*   **Tuning `alpha_c` (Cause Penalty, Default: 500m):** This is a critical parameter for ensuring causally pure clusters. It acts as a "spatial penalty" for mixing causes.
    *   **Action:** Inspect the final attributed maps. If you see clusters that illogically group different causes (e.g., a fire and a biotic event that are nearby but not overlapping), **increase `alpha_c`** (e.g., to 750m or 1000m). This makes it "harder" for different causes to mix. If you find that related but differently-labeled events (e.g., a confirmed 'Storm' and a Senf&Seidl 'Storm/Biotic' polygon for the same event) are being incorrectly split, you could **decrease `alpha_c`**.
*   **Tuning `alpha_t` (Temporal Scaling, Default: 10000):** This determines the importance of time relative to space. The default value makes space significantly more important than time. This is generally suitable for many forest disturbances. You would typically not need to tune this unless you have strong reason to believe that temporal precision is more important than spatial precision for your specific events.

### Cluster Size (`hdbscan_*_cluster_size_*`, `hdbscan_*_samples_*`)

*   **Concept:** These parameters control the minimum size and density of a valid cluster. The pipeline uses a robust method where the final `min_cluster_size` is the maximum of an absolute value and a relative fraction of the community size.
*   **Tuning Strategy:** These parameters are best tuned by observing the output.
    *   Are too many events being classified as "noise" (cluster ID -1) in the final output? This means the density criteria are too strict. **Decrease `hdbscan_min_cluster_size_abs`** (e.g., from 6 to 4) and **`hdbscan_min_samples_abs`** (e.g., from 2 to 1, though 1 can be risky).
    *   Are clearly separate disturbance events being merged into a single cluster? The criteria may be too loose. **Increase** the `min_cluster_size` and `min_samples` values.
    *   **Action:** Focus on the absolute values (`_abs`) for tuning. The relative values (`_rel`) help the parameters adapt to different community sizes and should generally be left alone.

---

## 4. Tuning Attribution and Voting

These parameters control the final voting logic.

### Dataset Reliability (`reliability`)

*   **Concept:** This is a dictionary of expert-defined scores reflecting your confidence in each dataset.
*   **Tuning Strategy:** This is purely driven by expert knowledge.
    *   Assign 1.0 to your most trusted "ground truth" sources (e.g., `firepolygons`).
    *   Assign lower values to datasets that are known to be noisy, have lower spatial precision, or are model-based (e.g., `cdi`).
    *   **Action:** Sit down with domain experts and assign a value to each dataset. This is a foundational step.

### Senf & Seidl Self-Vote (`senf_self_vote_factor`)

*   **Concept:** This parameter (Default: 0.3) sets the weight of a Senf & Seidl polygon's *own* original label during the voting process.
*   **Tuning Strategy:** This controls how much you trust the original S&S label versus the evidence from neighboring events.
    *   If you generally trust the auxiliary data more than the S&S map, keep this value low (0.1-0.3).
    *   If you believe the S&S map is highly accurate and should only be overturned by strong external evidence, increase this value (0.5-0.7).
    *   **Action:** Evaluate the final attributions for a set of well-known events. If the attribution is flipping to an incorrect class due to a single noisy neighbor, consider increasing the self-vote factor.

*Last updated: 2024-05-24* 