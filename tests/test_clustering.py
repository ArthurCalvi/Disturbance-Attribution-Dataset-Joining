import unittest
import geopandas as gpd
import networkx as nx
from src.join_datasets.utils import compute_similarity_matrix, get_cluster_v2

class TestClustering(unittest.TestCase):
    def setUp(self):
        # load small excerpt dataset
        path = 'excerpts/excerpt_simplified_CDI_2012_2023_EPSG2154_FR.parquet'
        self.data = gpd.read_parquet(path).head(10).copy()
        self.data['dataset'] = 'cdi'
        self.data['cause'] = self.data['class']

    def test_louvain_communities(self):
        # Build simple graph based on centroid distance
        centroids = self.data.geometry.centroid
        G = nx.Graph()
        for idx, pt in enumerate(centroids):
            G.add_node(idx)
        coords = list(centroids)
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = coords[i].distance(coords[j])
                if dist < 1000:
                    G.add_edge(i, j, weight=1 / (1 + dist))
        communities = nx.community.louvain_communities(G, seed=0, weight='weight')
        # ensure all nodes are assigned to a community
        covered = set().union(*communities)
        self.assertEqual(len(covered), len(self.data))
        self.assertGreaterEqual(len(communities), 1)

    def test_hdbscan_clustering(self):
        dtypes = {'cdi': 'polygon'}
        final_weighting_dict = {
            'cdi': {c: {'spatial': (lambda x: 1.0), 'temporal': (lambda x: 1.0)} for c in self.data['cause'].unique()}
        }
        sim, _ = compute_similarity_matrix(self.data, dtypes, {}, final_weighting_dict, weights=[1, 1])
        dclass = {'cdi': {c: {'drought': 1.0} for c in self.data['cause'].unique()}}
        clusters, _, labels = get_cluster_v2(
            self.data.copy(),
            sim,
            final_weighting_dict,
            {'cdi': 1.0},
            dclass,
            method='HDBSCAN',
            method_kwargs={'min_cluster_size': 2}
        )
        self.assertEqual(len(labels), len(self.data))
        self.assertGreaterEqual(len(clusters), 1)

if __name__ == '__main__':
    unittest.main()
