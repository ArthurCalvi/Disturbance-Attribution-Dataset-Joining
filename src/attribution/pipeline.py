from __future__ import annotations



from dataclasses import dataclass, field

from typing import Dict, Iterable, List, Tuple, Optional, Any
from datetime import datetime

import geopandas as gpd
import networkx as nx
import numpy as np


from tqdm.auto import tqdm
import logging
from pathlib import Path
import pickle
import pandas as pd

try:  # hdbscan is mandatory for clustering
    import hdbscan  # type: ignore
except Exception as exc:  # pragma: no cover - ensure clear error message
    raise ImportError(
        "hdbscan package is required for the attribution pipeline"
    ) from exc

logger = logging.getLogger(__name__)

@dataclass
class AttributionParams:
    """Parameters controlling the attribution pipeline."""

    spatial_half_life: float = 300.0  # Reduced from 1000.0 meters
    temporal_half_life: float = 180.0  # days
    lambda_intra: float = 0.5
    louvain_resolution: float = 1.0
    alpha_t: float = 10000.0
    alpha_c: float = 500.0


DEFAULT_RELIABILITY: Dict[str, float] = {
    "firepolygons": 1.0,
    "hm": 0.9,
    "forms": 0.8,
    "senfseidl": 0.7,
    "cdi": 0.5,
}


class Attribution:
    """Graph based disturbance attribution."""

    def __init__(
        self,
        gdf_dict: Dict[str, gpd.GeoDataFrame],
        params: AttributionParams | None = None,
        reliability: Dict[str, float] | None = None,
        temp_dir: Optional[str | Path] = None,
        force_recompute_prepared_data: bool = False,
        force_rebuild_graph: bool = False,
        force_redetect_communities: bool = False,
    ) -> None:
        self.params = params or AttributionParams()
        self.reliability = reliability or DEFAULT_RELIABILITY
        
        self.temp_dir = Path(temp_dir) if temp_dir else None
        # Store force recompute flags
        self.force_recompute_prepared_data_flag = force_recompute_prepared_data
        self.force_rebuild_graph_flag = force_rebuild_graph
        self.force_redetect_communities_flag = force_redetect_communities

        if self.temp_dir:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Temporary cache directory set to: {self.temp_dir}")

        # --- Caching for _prepare_data ---
        prepared_data_path = self.temp_dir / "temp_prepared_data.parquet" if self.temp_dir else None
        # Use the stored flag for prepared data as well for consistency (though it was directly passed earlier)
        if not self.force_recompute_prepared_data_flag and prepared_data_path and prepared_data_path.exists():
            try:
                logger.info(f"Loading cached prepared data from {prepared_data_path}...")
                self.data = gpd.read_parquet(prepared_data_path)
                logger.info(f"Successfully loaded cached prepared data: {len(self.data)} records.")
            except Exception as e:
                logger.warning(f"Could not load cached prepared data from {prepared_data_path}: {e}. Re-calculating.")
                self.data = self._prepare_data(gdf_dict) # Fallback
        else:
            self.data = self._prepare_data(gdf_dict)
        # --- End Caching for _prepare_data ---
        
        self.graph = nx.Graph() # Initialize graph, might be overwritten by cache
        self.communities: List[set[int]] = [] # Initialize, might be overwritten
        self.cluster_labels: Dict[int, int] = {}

        logger.info("Attribution initialised with %d features", len(self.data))

    def _prepare_data(self, gdfs: Dict[str, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        logger.info("Executing _prepare_data...")
        frames = []
        for name, gdf in gdfs.items():
            df = gdf.copy()
            df["dataset"] = name
            if "start_date" in df.columns:
                df["start_date"] = gpd.pd.to_datetime(df["start_date"], errors="coerce")
                if df["start_date"].dt.tz is not None:
                    df["start_date"] = df["start_date"].dt.tz_convert('UTC').dt.tz_localize(None)
                else:
                    # If already naive, ensure it's treated as such consistently
                    df["start_date"] = df["start_date"].dt.tz_localize(None)


            if "end_date" in df.columns:
                df["end_date"] = gpd.pd.to_datetime(df["end_date"], errors="coerce")
                if df["end_date"].dt.tz is not None:
                    df["end_date"] = df["end_date"].dt.tz_convert('UTC').dt.tz_localize(None)
                else:
                     # If already naive, ensure it's treated as such consistently
                    df["end_date"] = df["end_date"].dt.tz_localize(None)

            if "year" in df.columns and "start_date" not in df.columns:
                # This will create naive datetime objects
                # df["start_date"] = gpd.pd.to_datetime(df["year"], format="%Y")
                # # Ensure end_date is also naive if derived from year
                # df["end_date"] = df["start_date"] 
                df["start_date"] = gpd.pd.to_datetime(df["year"].astype(str) + '-06-01', format='%Y-%m-%d', errors='coerce')
                df["end_date"] = df["start_date"] # Set end_date to be the same as start_date (June 1st)

            if "start_date" in df.columns and "end_date" not in df.columns:
                 # Ensure end_date is naive if derived from start_date
                df["end_date"] = df["start_date"]
            
            # Ensure mid_date is calculated from consistently naive UTC dates
            if "start_date" in df.columns and "end_date" in df.columns:
                # Coerce to datetime again just in case some operations changed type, and ensure naivety
                start_dates = gpd.pd.to_datetime(df["start_date"], errors="coerce").dt.tz_localize(None)
                end_dates = gpd.pd.to_datetime(df["end_date"], errors="coerce").dt.tz_localize(None)
                
                # Handle NaT values explicitly to avoid issues with timedelta calculation
                valid_dates_mask = start_dates.notna() & end_dates.notna()
                df["mid_date"] = gpd.pd.NaT # Initialize with NaT
                df.loc[valid_dates_mask, "mid_date"] = start_dates[valid_dates_mask] + (end_dates[valid_dates_mask] - start_dates[valid_dates_mask]) / 2
            else:
                # Fallback or error if essential date columns are missing after processing
                logger.warning(f"Dataset {name} missing standardized start_date or end_date for mid_date calculation.")
                df["mid_date"] = gpd.pd.NaT

            # Log date column statistics for the current dataset before appending
            logger.debug(f"--- Date Column Stats for dataset: {name} ---")
            for col_name in ["year", "start_date", "end_date", "mid_date"]:
                if col_name in df.columns:
                    total_count = len(df)
                    nan_count = df[col_name].isna().sum()
                    if total_count > 0:
                        nan_percentage = (nan_count / total_count) * 100
                        logger.debug(f"  Column '{col_name}': {nan_count} NaNs ({nan_percentage:.2f}%) out of {total_count} records.")
                    else:
                        logger.debug(f"  Column '{col_name}': Dataset is empty.")
                else:
                    logger.debug(f"  Column '{col_name}': Not present in this dataset.")
            logger.debug(f"--------------------------------------------")

            frames.append(df)
        all_data = gpd.GeoDataFrame(gpd.pd.concat(frames, ignore_index=True), crs=frames[0].crs)
        all_data["uid"] = all_data.index

        logger.debug("Prepared dataframe with columns: %s", list(all_data.columns))
        
        # --- Save prepared_data if temp_dir is set ---
        if self.temp_dir:
            prepared_data_path = self.temp_dir / "temp_prepared_data.parquet"
            try:
                logger.debug(f"Caching prepared_data to {prepared_data_path}...")
                all_data.to_parquet(prepared_data_path)
                logger.debug("Successfully cached prepared_data.")
            except Exception as e:
                logger.error(f"Could not cache prepared_data to {prepared_data_path}: {e}")
        # --- End Save ---

        return all_data

    def log_parameters(self) -> None:
        """Logs the parameters used for the attribution."""
        logger.debug("--- Attribution Parameters ---")
        if self.params:
            for param, value in self.params.__dict__.items():
                logger.debug(f"  {param}: {value}")
        else:
            logger.debug("  No specific AttributionParams provided (using defaults directly in code perhaps).")
        
        logger.debug("--- Dataset Reliability ---")
        if self.reliability:
            for dataset, rel_value in self.reliability.items():
                logger.debug(f"  {dataset}: {rel_value}")
        else:
            logger.debug("  No specific reliability dictionary provided.")
        logger.debug("-----------------------------")

    # ------------------------------------------------------------------
    # Edge building
    # ------------------------------------------------------------------
    def _temporal_distance(self, a: gpd.pd.Timestamp, b: gpd.pd.Timestamp) -> float:
        if gpd.pd.isna(a) or gpd.pd.isna(b):
            return np.inf
        delta = a - b
        return abs(delta.days)

    def _candidate_pairs(self) -> Iterable[Tuple[int, int]]:
        sindex = self.data.sindex
        for idx, geom in enumerate(self.data.geometry):
            bbox = geom.centroid.buffer(1500).bounds # Reduced from 6000
            candidates = list(sindex.intersection(bbox))
            for j in candidates:
                if j <= idx:
                    continue
                dt = self._temporal_distance(
                    self.data.loc[idx, "mid_date"],
                    self.data.loc[j, "mid_date"],
                )
                if dt > 720:
                    continue
                yield idx, j

    def _edge_weight(self, idx: int, j: int) -> float:
        a = self.data.loc[idx]
        b = self.data.loc[j]
        ds = a.geometry.distance(b.geometry)
        if ds > 1500: # Reduced from 6000
            return 0.0
        w_s = np.exp(-ds / self.params.spatial_half_life)
        dt = self._temporal_distance(a.mid_date, b.mid_date)
        w_t = np.exp(-dt / self.params.temporal_half_life)
        w_prox = w_s * w_t
        r = (self.reliability.get(a.dataset, 0.5) + self.reliability.get(b.dataset, 0.5)) / 2
        lam = self.params.lambda_intra if a.dataset == b.dataset else 1.0
        return w_prox * r * lam

    def build_graph(self) -> None:
        # --- Caching for build_graph ---
        graph_path = self.temp_dir / "temp_graph.graphml" if self.temp_dir else None
        graph_data_path = self.temp_dir / "temp_graph_data.parquet" if self.temp_dir else None

        if not self.force_rebuild_graph_flag and graph_path and graph_path.exists() and graph_data_path and graph_data_path.exists():
            try:
                logger.info(f"Loading cached graph from {graph_path} and associated data from {graph_data_path}...")
                self.graph = nx.read_graphml(graph_path)
                # Ensure node types are correct (GraphML saves them as strings)
                self.graph = nx.relabel_nodes(self.graph, {node_str: int(node_str) for node_str in self.graph.nodes()})
                
                # Load the data that was current when this graph was saved
                # This is important if _candidate_pairs or other graph-related steps modify self.data
                # For now, assuming self.data loaded by __init__ (or its cache) is sufficient if not modified by graph steps.
                # If graph building logic adds/modifies columns in self.data that are crucial for later stages,
                # that version of self.data MUST be saved alongside the graph and reloaded here.
                # The current `temp_prepared_data.parquet` might suffice if self.data isn't altered by graph building itself.
                # For safety, let's assume we save/load a specific version of data with the graph.
                self.data = gpd.read_parquet(graph_data_path)

                logger.info(f"Successfully loaded cached graph ({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges) and associated data.")
                logger.info("Skipping graph building step.")
                return # Skip building
            except Exception as e:
                logger.warning(f"Could not load cached graph/data from {graph_path}/{graph_data_path}: {e}. Re-building graph.")
        # --- End Caching for build_graph ---

        logger.info("Building graph with %d nodes", len(self.data))
        for idx in tqdm(self.data.index, desc="nodes"):
            self.graph.add_node(idx)
        for i, j in tqdm(self._candidate_pairs(), desc="edges"):
            weight = self._edge_weight(i, j)
            if weight > 0:
                self.graph.add_edge(i, j, weight=weight)
        
        # --- Save graph if temp_dir is set ---
        if self.temp_dir:
            graph_path_out = self.temp_dir / "temp_graph.graphml"
            graph_data_path_out = self.temp_dir / "temp_graph_data.parquet"
            try:
                logger.debug(f"Caching graph to {graph_path_out} and current data to {graph_data_path_out}...")
                nx.write_graphml(self.graph, graph_path_out)
                self.data.to_parquet(graph_data_path_out) # Save the current state of self.data
                logger.debug("Successfully cached graph and associated data.")
            except Exception as e:
                logger.error(f"Could not cache graph or data: {e}")
        # --- End Save ---

    # ------------------------------------------------------------------
    # Louvain / HDBSCAN
    # ------------------------------------------------------------------
    def detect_communities(self) -> None:
        # --- Caching for detect_communities ---
        data_with_communities_path = self.temp_dir / "temp_data_with_communities.parquet" if self.temp_dir else None
        
        if not self.force_redetect_communities_flag and data_with_communities_path and data_with_communities_path.exists():
            try:
                logger.info(f"Loading cached data with communities from {data_with_communities_path}...")
                self.data = gpd.read_parquet(data_with_communities_path)
                
                if "community_id" not in self.data.columns:
                    raise ValueError("Cached data does not contain 'community_id' column.")

                # Reconstruct self.communities list from the loaded data
                self.communities = []
                if self.data["community_id"].notna().any(): # Check if there are any non-NaN community_ids
                    # Group by community_id and collect the original indices (uid)
                    # Ensure community_id is treated as integer if possible, handle NaNs
                    self.data["community_id"] = gpd.pd.to_numeric(self.data["community_id"], errors='coerce').astype('Int64')
                    grouped_communities = self.data[self.data["community_id"].notna()].groupby("community_id")["uid"].apply(set)
                    self.communities = [members for _, members in sorted(grouped_communities.items())]
                
                logger.info(f"Successfully loaded cached data with communities. Found {len(self.communities)} communities from cache.")
                logger.info("Skipping community detection step.")
                return # Skip detection
            except Exception as e:
                logger.warning(f"Could not load cached data with communities from {data_with_communities_path}: {e}. Re-detecting communities.")
        # --- End Caching ---

        logger.info("Detecting Louvain communities")
        self.communities = list(
            nx.community.louvain_communities(
                self.graph,
                resolution=self.params.louvain_resolution,
                weight="weight",
            )
        )
        logger.debug("%d communities found", len(self.communities))
        
        for cid, members in enumerate(self.communities):
            for m in members:
                self.data.loc[m, "community_id"] = cid
        
        # --- Save data with communities if temp_dir is set ---
        if self.temp_dir:
            data_comm_path_out = self.temp_dir / "temp_data_with_communities.parquet"
            try:
                logger.debug(f"Caching data with communities to {data_comm_path_out}...")
                self.data.to_parquet(data_comm_path_out)
                logger.debug("Successfully cached data with communities.")
            except Exception as e:
                logger.error(f"Could not cache data with communities: {e}")
        # --- End Save ---

    def _hdbscan_cluster(self, members: Iterable[int], cluster_base: int) -> int:
        if hdbscan is None:  # pragma: no cover - optional dependency
            raise ImportError("hdbscan package is required for clustering")

        original_member_list = list(members) # Keep a list for logging/iteration
        if not original_member_list:
            logger.warning(f"Skipping HDBSCAN for an empty community. Cluster_base: {cluster_base}")
            return cluster_base

        df = self.data.loc[original_member_list].copy()
        
        if df.empty:
            logger.warning(f"DataFrame for HDBSCAN is empty after selecting members. Community members: {original_member_list}. Cluster_base: {cluster_base}")
            for node_idx in original_member_list:
                if node_idx in self.data.index: # Should always be true if original_member_list came from self.data
                    self.cluster_labels[node_idx] = -1
                    self.data.loc[node_idx, "hdb_id"] = -1
            return cluster_base

        # --- Handle NaNs in mid_date ---
        if df["mid_date"].isna().any():
            logger.warning(f"Community (base {cluster_base}, {len(original_member_list)} members) has NaT in 'mid_date'. Attempting to filter.")
            df_before_nan_filter_len = len(df)
            df.dropna(subset=["mid_date"], inplace=True)
            logger.info(f"Filtered 'mid_date' NaNs: {df_before_nan_filter_len} -> {len(df)} members for community (base {cluster_base}).")

        # Check if df is too small or empty after mid_date NaN removal
        # Use a practical minimum size, e.g., 3, as HDBSCAN needs some points. min_cluster_size will be calculated later.
        MIN_POINTS_FOR_HDBSCAN = 2
        if len(df) < MIN_POINTS_FOR_HDBSCAN:
            logger.warning(f"Community (base {cluster_base}) has less than {MIN_POINTS_FOR_HDBSCAN} members after 'mid_date' NaN filtering ({len(df)} members). Assigning all original members to noise.")
            for node_idx in original_member_list:
                self.cluster_labels[node_idx] = -1
                self.data.loc[node_idx, "hdb_id"] = -1
            return cluster_base
        # --- End Handle NaNs in mid_date ---

        df["x"] = df.geometry.centroid.x
        df["y"] = df.geometry.centroid.y
        
        median_time = df["mid_date"].median()
        if gpd.pd.isna(median_time):
            logger.warning(f"Median_time is NaT for community (base {cluster_base}, {len(df)} members). This can happen if all mid_dates were NaT or problematic. Assigning noise.")
            for node_idx in original_member_list: # Assign noise to original members
                self.cluster_labels[node_idx] = -1
                self.data.loc[node_idx, "hdb_id"] = -1
            return cluster_base

        df["t"] = (df["mid_date"] - median_time).dt.days / self.params.alpha_t
        df["cause"] = df["class"].astype("category").cat.codes
        
        feature_cols = ["x", "y", "t", "cause"]
        X = df[feature_cols].to_numpy()

        # Check for NaNs/Infs in X right before HDBSCAN
        if not np.all(np.isfinite(X)):
            logger.error(f"Non-finite values (NaN or Inf) found in X for HDBSCAN for community (base {cluster_base}). Assigning noise to original members. Features: {feature_cols}")
            logger.debug(f"Problematic X sample for community (base {cluster_base}):\n{df[feature_cols][~np.all(np.isfinite(X), axis=1)].head()}")
            for node_idx in original_member_list:
                self.cluster_labels[node_idx] = -1
                self.data.loc[node_idx, "hdb_id"] = -1
            return cluster_base

        if X.shape[0] == 0: # Should be caught by len(df) < MIN_POINTS_FOR_HDBSCAN, but as a safeguard
            logger.warning(f"Input array X for HDBSCAN is empty (0 rows) for community (base {cluster_base}). Assigning noise to original members.")
            for node_idx in original_member_list:
                self.cluster_labels[node_idx] = -1
                self.data.loc[node_idx, "hdb_id"] = -1
            return cluster_base

        min_cluster_size = max(6, int(0.05 * len(df)))
        min_samples = max(2, int(0.02 * len(df)))
        
        # Ensure min_cluster_size is not greater than the number of samples in X
        if min_cluster_size > X.shape[0]:
            min_cluster_size = X.shape[0]
            logger.debug(f"Adjusted min_cluster_size to {min_cluster_size} (number of samples) for community (base {cluster_base}).")
        if min_samples is not None and min_samples > X.shape[0]: # min_samples can be None for HDBSCAN*
             min_samples = max(1, X.shape[0] // 2) # Or X.shape[0] if very small
             logger.debug(f"Adjusted min_samples to {min_samples} for community (base {cluster_base}).")


        logger.debug(f"Running HDBSCAN for community (base {cluster_base}): {len(original_member_list)} original members, {X.shape[0]} points in X.")
        logger.debug(f"  HDBSCAN params: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
        logger.debug(f"  X sample (first 5 rows) for community (base {cluster_base}):\n{X[:5]}")


        def metric(a: np.ndarray, b: np.ndarray) -> float:
            # spatial component uses x, y, t (indices 0, 1, 2)
            spatial_diff = a[:3] - b[:3]
            spatial = np.sqrt(np.sum(spatial_diff**2))
            # cause component uses cause code (index 3)
            cause_pen = self.params.alpha_c if a[3] != b[3] else 0.0
            return spatial + cause_pen
        
        # Check if X.shape[0] is less than min_samples. Some HDBSCAN versions might error.
        # Or if min_samples becomes 0 or too small.
        # The hdbscan library usually handles small sample sizes by not finding clusters.
        # The ValueError "Invalid shape in axis 0: 0" usually points to an empty array being processed at C-level.

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples, # If min_samples is 0 from calc, might be an issue. Max(1, ...) ensures it's at least 1.
            metric=metric,
            allow_single_cluster=True # Might help with very small communities
        )
        labels = clusterer.fit_predict(X)

        logger.debug(f"HDBSCAN cluster labels for community (base {cluster_base}): {np.unique(labels)}")

        # Map labels back to original data indices (from df.index, which are the filtered members)
        for node_idx, lab in zip(df.index, labels):
            if lab >= 0:
                label = cluster_base + lab
            else:
                label = -1 # Noise point
            self.cluster_labels[node_idx] = label
            self.data.loc[node_idx, "hdb_id"] = label
        
        # For original members who were filtered out (e.g. due to NaT mid_date and not in df.index anymore)
        # assign them to noise if they haven't got a label.
        # This is covered if all original_member_list are assigned noise if df becomes too small.
        # If df was processed, only df.index members get labels here.
        # Ensure all original_member_list members get some hdb_id, default to -1 if filtered out before clustering.
        processed_indices = set(df.index)
        for node_idx in original_member_list:
            if node_idx not in processed_indices:
                if node_idx not in self.cluster_labels: # If not already assigned noise
                    logger.debug(f"Assigning noise to member {node_idx} (filtered pre-HDBSCAN) of community (base {cluster_base}).")
                    self.cluster_labels[node_idx] = -1
                    self.data.loc[node_idx, "hdb_id"] = -1
        
        # Calculate the next cluster_base
        # If labels contains only -1 (all noise), labels.max() would be -1.
        # next_cluster_base_increment should be 0 if no actual clusters were found.
        next_cluster_base_increment = 0
        if labels.size > 0 and labels.max() >=0: # Check if any valid clusters were formed
            next_cluster_base_increment = labels.max() + 1
            
        return cluster_base + next_cluster_base_increment

    def run_hdbscan(self) -> None:

        logger.info("Running HDBSCAN on communities")
        
        # Log overall date column statistics for self.data before HDBSCAN
        logger.info("--- Pre-HDBSCAN Date Column Stats for self.data ---")
        for col_name in ["year", "start_date", "end_date", "mid_date"]:
            if col_name in self.data.columns:
                total_count = len(self.data)
                nan_count = self.data[col_name].isna().sum()
                if total_count > 0:
                    nan_percentage = (nan_count / total_count) * 100
                    logger.info(f"  Column '{col_name}': {nan_count} NaNs ({nan_percentage:.2f}%) out of {total_count} records.")
                else:
                    logger.info(f"  Column '{col_name}': self.data is empty.")
            else:
                logger.info(f"  Column '{col_name}': Not present in self.data.")
        logger.info("----------------------------------------------------")

        next_label = 0
        for members in tqdm(self.communities, desc="hdbscan"):
            next_label = self._hdbscan_cluster(members, next_label)

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------
    def _votes_for_cluster(self, members: Iterable[int]) -> Dict[str, float]:
        votes: Dict[str, float] = {}
        for idx in members:
            row = self.data.loc[idx]
            # Ensure 'class' and 'dataset' columns exist for the row
            cls = row.get("class", "unknown_class") # Default if 'class' is missing
            dataset_name = row.get("dataset", "unknown_dataset") # Default if 'dataset' is missing
            
            w = self.reliability.get(dataset_name, 0.5) # Use reliability of the member's dataset
            votes[cls] = votes.get(cls, 0.0) + w
        return votes

    def attribute(self) -> gpd.GeoDataFrame: # Changed return type
        logger.info("Computing attribution votes for Senf&Seidl events")

        if "hdb_id" not in self.data.columns and "community_id" not in self.data.columns:
            logger.error("Neither 'hdb_id' nor 'community_id' found in data. Cannot perform attribution. Returning data as is.")
            return self.data
        
        primary_group_field = "hdb_id" if "hdb_id" in self.data.columns else "community_id"
        logger.info(f"Using '{primary_group_field}' as the primary field for attribution grouping strategy.")

        # Initialize probability columns for all unique disturbance classes found in the data
        all_disturbance_classes = [cls for cls in self.data["class"].unique() if pd.notna(cls)]
        for cls_name in all_disturbance_classes:
            self.data[f"prob_{cls_name}"] = 0.0  # Initialize with 0.0

        # --- Pre-calculate base votes for all clusters ---
        hdb_cluster_votes_map: Dict[Any, Dict[str, float]] = {}
        community_cluster_votes_map: Dict[Any, Dict[str, float]] = {}

        if "hdb_id" in self.data.columns:
            logger.info("Pre-calculating base votes for HDBSCAN clusters (hdb_id)...")
            # Filter out noise/NaN hdb_ids for pre-calculation
            valid_hdb_clusters = self.data[self.data["hdb_id"].notna() & (self.data["hdb_id"] != -1)]
            if not valid_hdb_clusters.empty:
                grouped_by_hdb = valid_hdb_clusters.groupby("hdb_id")
                for hdb_id_val, group_df in tqdm(grouped_by_hdb, desc="Votes for hdb_id clusters", leave=False):
                    hdb_cluster_votes_map[hdb_id_val] = self._votes_for_cluster(group_df.index)
            logger.info(f"Base votes pre-calculated for {len(hdb_cluster_votes_map)} HDBSCAN clusters.")

        if "community_id" in self.data.columns:
            logger.info("Pre-calculating base votes for Louvain communities (community_id)...")
            # For community_id, NaN is the main indicator of no community if a node wasn't assigned.
            # Louvain communities are typically 0-indexed if assigned.
            valid_community_clusters = self.data[self.data["community_id"].notna()]
            if not valid_community_clusters.empty:
                grouped_by_community = valid_community_clusters.groupby("community_id")
                for community_id_val, group_df in tqdm(grouped_by_community, desc="Votes for community_id clusters", leave=False):
                    community_cluster_votes_map[community_id_val] = self._votes_for_cluster(group_df.index)
            logger.info(f"Base votes pre-calculated for {len(community_cluster_votes_map)} Louvain communities.")
        
        # Filter for Senf&Seidl polygons to iterate over
        senf_indices = self.data[self.data["dataset"] == "senfseidl"].index
        
        if len(senf_indices) == 0:
            logger.warning("No Senf&Seidl events found in the data. Attribution step will not compute probabilities.")
            return self.data

        logger.info(f"Starting attribution for {len(senf_indices)} Senf&Seidl events...")

        for idx in tqdm(senf_indices, desc="Attributing Senf&Seidl"):
            row = self.data.loc[idx]
            current_votes: Dict[str, float] = {}
            processed_with_cluster_votes = False

            # Try primary group field (hdb_id if available and "hdb_id" is the chosen primary)
            if primary_group_field == "hdb_id" and "hdb_id" in self.data.columns:
                hdb_id_val = row.get("hdb_id")
                if pd.notna(hdb_id_val) and hdb_id_val != -1 and hdb_id_val in hdb_cluster_votes_map:
                    current_votes = hdb_cluster_votes_map[hdb_id_val].copy()
                    processed_with_cluster_votes = True
            
            # If not processed by hdb_id (either it was noise, NaN, or primary was community_id), 
            # try community_id (if available).
            if not processed_with_cluster_votes and "community_id" in self.data.columns:
                community_id_val = row.get("community_id")
                # Ensure community_id_val is a valid key type (not NaN) and exists in the map
                if pd.notna(community_id_val) and community_id_val in community_cluster_votes_map:
                    current_votes = community_cluster_votes_map[community_id_val].copy()
                    processed_with_cluster_votes = True
            
            # Add the Senf&Seidl self-vote component
            senf_event_class = row.get("class")
            r_s = self.reliability.get("senfseidl", 0.7) # Senf&Seidl reliability

            if pd.notna(senf_event_class):
                if processed_with_cluster_votes: # Base votes from cluster are in current_votes
                    current_votes[senf_event_class] = current_votes.get(senf_event_class, 0.0) + (0.3 * r_s)
                else: # No base votes from cluster (isolated/noise), so self-vote is the only component
                    current_votes[senf_event_class] = (0.3 * r_s)
                    logger.debug(f"Senf&Seidl event {idx} (class: {senf_event_class}) is isolated or in a non-voting/fallback cluster. Using self-vote component only for its class.")
            
            # Normalize votes to probabilities
            total_vote_sum = sum(current_votes.values())
            if total_vote_sum > 0:
                for cause, vote_sum in current_votes.items():
                    if pd.notna(cause): # Ensure cause is a valid name
                        self.data.loc[idx, f"prob_{cause}"] = vote_sum / total_vote_sum
            elif pd.notna(senf_event_class) and not current_votes and not processed_with_cluster_votes : # Only self-vote was possible but resulted in 0 (e.g. r_s=0)
                 logger.debug(f"Senf&Seidl event {idx} (class: {senf_event_class}) had zero total_vote_sum from self-vote component, possibly due to reliability or class issues.")


        logger.info("Attribution probabilities computed and added to the GeoDataFrame for Senf&Seidl events.")
        return self.data

