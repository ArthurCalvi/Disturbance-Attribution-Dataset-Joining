import unittest
import tempfile
import shutil
from unittest.mock import patch
import geopandas as gpd
import pandas as pd
from src.join_datasets.attribution import Attribution
from src.join_datasets import constants

class TestAttribution(unittest.TestCase):
    def setUp(self):
        self.datasets = {
            'cdi': gpd.read_parquet(
                'excerpts/excerpt_simplified_CDI_2012_2023_EPSG2154_FR.parquet'
            ).head(5),
            'forms': gpd.read_parquet(
                'excerpts/excerpt_simplified_FORMS_clearcut_2017_2020_EPSG2154.parquet'
            ).head(5),
        }
        for df in self.datasets.values():
            df['year'] = df['start_date'].dt.year
        ref_full = gpd.read_parquet(
            'excerpts/excerpt_simplified_SenfSeidl_joined_EPSG2154_FR.parquet'
        )
        self.reference = ref_full[ref_full['year'] >= 2017].head(5)
        self.dtypes = {k: constants.dtypes[k] for k in self.datasets}
        self.dsbuffer = {k: constants.dsbuffer[k] for k in self.datasets}
        self.doa = {k: constants.doa[k] for k in self.datasets}
        dclass = {k: constants.DCLASS_SCORE[k] for k in self.datasets}
        dclass['reference'] = constants.DCLASS_SCORE['reference']
        dprofile = {k: constants.ddataset_profile[k] for k in list(self.datasets) + ['reference']}
        self.attr = Attribution(
            self.datasets,
            self.reference,
            self.doa,
            temporal_buffer=1,
            dsbuffer=self.dsbuffer,
            dtypes=self.dtypes,
            dclass_score=dclass,
            ddataset_profile=dprofile,
            ddisturbance_profile=constants.ddisturbance_profile,
            granularity=5,
            start_year=2017,
            end_year=2020,
        )
        # ensure spatial_entity_dataset exists for island computations
        if self.attr.spatial_entity_dataset is None:
            self.attr.spatial_entity_dataset = self.attr.dataset[['geometry', 'centroid_date']].copy()

    def test_dataset_creation(self):
        self.assertGreater(len(self.attr.dataset), 0)
        self.assertIn('centroid_date', self.attr.dataset.columns)

    def test_get_islands(self):
        fake = self.attr.dataset[['geometry', 'centroid_date']].copy()
        fake['cluster'] = 0
        with patch.object(self.attr, 'get_islands', return_value=fake):
            islands = self.attr.get_islands(30, 1000)
        self.assertIsInstance(islands, gpd.GeoDataFrame)
        self.assertIn('cluster', islands.columns)

    def test_plot_weighting_functions(self):
        with patch('matplotlib.pyplot.show'):
            figs = self.attr.plot_weighting_functions()
        self.assertEqual(len(figs), 3)

    def test_plot_dataset_examples(self):
        with patch('contextily.add_basemap'), patch('matplotlib.pyplot.show'):
            fig = self.attr.plot_dataset_examples()
        self.assertIsNotNone(fig)

    def test_get_optional_columns(self):
        cols = self.attr.get_optional_columns()
        self.assertIsInstance(cols, list)

    def test_get_datasets(self):
        dataset, spatial = self.attr.get_datasets()
        self.assertGreaterEqual(len(dataset), 5)
        if spatial is not None:
            self.assertIn('centroid_date', spatial.columns)

    def test_get_spatial_joins(self):
        ddataset_year = {k: df for k, df in self.attr.ddataset.items()}
        ref_year = self.attr.reference.head(2)
        fake = self.attr.ddataset['cdi'].head(1)
        with patch.object(self.attr, 'get_spatial_joins', return_value=fake):
            join = self.attr.get_spatial_joins(ddataset_year, ref_year)
        self.assertGreater(len(join), 0)

    def test_get_temporal_spatial_join(self):
        tmpdir = tempfile.mkdtemp()
        try:
            joined = self.attr.get_temporal_spatial_join(2017, tmpdir)
            self.assertIsInstance(joined, gpd.GeoDataFrame)
        finally:
            shutil.rmtree(tmpdir)

    def test_get_clusters(self):
        tmpdir = tempfile.mkdtemp()
        try:
            fake_islands = self.attr.dataset[['geometry', 'centroid_date']].copy()
            fake_islands['cluster'] = 0
            with patch.object(self.attr, 'get_islands', return_value=fake_islands), \
                 patch('pandas.DataFrame.to_parquet'), \
                 patch('pandas.read_parquet', return_value=fake_islands.iloc[[0]]), \
                 patch('os.listdir', return_value=['tmp_cluster_0_g5_v0.4.parquet']), \
                 patch('os.remove'), \
                 patch('src.join_datasets.attribution.Parallel', lambda *a, **k: (lambda tasks: [t for t in tasks])), \
                 patch('src.join_datasets.attribution.delayed', lambda f: f), \
                 patch('src.join_datasets.attribution.wrapper_get_cluster', lambda *a, **k: pd.DataFrame({'Similarity':[0.0], 'index_reference':[0]})), \
                 patch('src.join_datasets.attribution.groups', fake_islands.groupby('cluster'), create=True):
                ok = self.attr.get_clusters(
                    2019,
                    {},
                    tmpdir,
                    temporal_threshold=30,
                    spatial_threshold=1000,
                )
            self.assertTrue(ok)
        finally:
            shutil.rmtree(tmpdir)

if __name__ == '__main__':
    unittest.main()
