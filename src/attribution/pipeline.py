from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
from datetime import datetime

import geopandas as gpd
import networkx as nx
import numpy as np
from tqdm.auto import tqdm
import logging

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

    spatial_half_life: float = 1000.0
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
    ) -> None:
        self.params = params or AttributionParams()
        self.reliability = reliability or DEFAULT_RELIABILITY
        self.data = self._prepare_data(gdf_dict)
        self.graph = nx.Graph()
        self.communities: List[set[int]] = []
        self.cluster_labels: Dict[int, int] = {}
        logger.info("Attribution initialised with %d features", len(self.data))

    def _prepare_data(self, gdfs: Dict[str, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        frames = []
        for name, gdf in gdfs.items():
            df = gdf.copy()
            df["dataset"] = name
            if "start_date" in df.columns:
                df["start_date"] = gpd.pd.to_datetime(df["start_date"], errors="coerce")
            if "end_date" in df.columns:
                df["end_date"] = gpd.pd.to_datetime(df["end_date"], errors="coerce")
            if "year" in df.columns and "start_date" not in df.columns:
                df["start_date"] = gpd.pd.to_datetime(df["year"], format="%Y")
                df["end_date"] = df["start_date"]
            if "end_date" not in df.columns:
                df["end_date"] = df["start_date"]
            df["mid_date"] = df["start_date"] + (df["end_date"] - df["start_date"]) / 2
            frames.append(df)
        all_data = gpd.GeoDataFrame(gpd.pd.concat(frames, ignore_index=True), crs=frames[0].crs)
        all_data["uid"] = all_data.index
        logger.debug("Prepared dataframe with columns: %s", list(all_data.columns))
        return all_data

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
            bbox = geom.centroid.buffer(6000).bounds
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
        if ds > 6000:
            return 0.0
        w_s = np.exp(-ds / self.params.spatial_half_life)
        dt = self._temporal_distance(a.mid_date, b.mid_date)
        w_t = np.exp(-dt / self.params.temporal_half_life)
        w_prox = w_s * w_t
        r = (self.reliability.get(a.dataset, 0.5) + self.reliability.get(b.dataset, 0.5)) / 2
        lam = self.params.lambda_intra if a.dataset == b.dataset else 1.0
        return w_prox * r * lam

    def build_graph(self) -> None:
        logger.info("Building graph with %d nodes", len(self.data))
        for idx in tqdm(self.data.index, desc="nodes"):
            self.graph.add_node(idx)
        for i, j in tqdm(self._candidate_pairs(), desc="edges"):
            weight = self._edge_weight(i, j)
            if weight > 0:
                self.graph.add_edge(i, j, weight=weight)

    # ------------------------------------------------------------------
    # Louvain / HDBSCAN
    # ------------------------------------------------------------------
    def detect_communities(self) -> None:
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

    def _hdbscan_cluster(self, members: Iterable[int], cluster_base: int) -> int:
        df = self.data.loc[list(members)].copy()
        df["x"] = df.geometry.centroid.x
        df["y"] = df.geometry.centroid.y
        median_time = df["mid_date"].median()
        df["t"] = (df["mid_date"] - median_time).dt.days / self.params.alpha_t
        df["cause"] = df["class"].astype("category").cat.codes
        X = df[["x", "y", "t", "cause"]].to_numpy()

        min_cluster_size = max(6, int(0.05 * len(df)))
        min_samples = max(2, int(0.02 * len(df)))

        def metric(a: np.ndarray, b: np.ndarray) -> float:
            spatial = np.sqrt(((a[:3] - b[:3]) ** 2).sum())
            cause_pen = self.params.alpha_c if a[3] != b[3] else 0.0
            return spatial + cause_pen

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
        )
        labels = clusterer.fit_predict(X)
        logger.debug("HDBSCAN cluster labels: %s", np.unique(labels))
        for node, lab in zip(df.index, labels):
            if lab >= 0:
                label = cluster_base + lab
            else:
                label = -1
            self.cluster_labels[node] = label
            self.data.loc[node, "hdb_id"] = label
        return cluster_base + (labels.max() + 1 if labels.max() >= 0 else 0)

    def run_hdbscan(self) -> None:
        logger.info("Running HDBSCAN on communities")
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
            cls = row.get("class", "unknown")
            w = self.reliability.get(row.dataset, 0.5)
            votes[cls] = votes.get(cls, 0.0) + w
        return votes

    def attribute(self) -> Dict[int, Dict[str, float]]:
        logger.info("Computing attribution votes")
        if "hdb_id" in self.data.columns:
            group_field = "hdb_id"
        else:
            group_field = "community_id"
        result = {}
        senf = self.data[self.data["dataset"] == "senfseidl"]
        for idx in tqdm(senf.index, desc="attribute"):
            row = self.data.loc[idx]
            cid = row[group_field]
            if cid == -1:
                cid = row.get("community_id")
            members = self.data[self.data[group_field] == cid].index
            votes = self._votes_for_cluster(members)
            # self vote 0.3 * r_S
            r_s = self.reliability.get("senfseidl", 0.7)
            votes[row["class"]] = votes.get(row["class"], 0.0) + 0.3 * r_s
            total = sum(votes.values())
            if total == 0:
                continue
            probs = {k: v / total for k, v in votes.items()}
            result[int(idx)] = probs
        return result

