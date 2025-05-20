import unittest
import geopandas as gpd
import pandas as pd
from src.join_datasets.attribution import Attribution
from src.join_datasets.utils import compute_similarity_matrix

class TestAttribution(unittest.TestCase):
    def setUp(self):
        cdi_path = 'excerpts/excerpt_simplified_CDI_2012_2023_EPSG2154_FR.parquet'
        forms_path = 'excerpts/excerpt_simplified_FORMS_clearcut_2017_2020_EPSG2154.parquet'
        self.cdi = gpd.read_parquet(cdi_path).head(5).copy()
        self.forms = gpd.read_parquet(forms_path).head(5).copy()
        for df, name in [(self.cdi, 'cdi'), (self.forms, 'forms')]:
            df['dataset'] = name
            df['start_date'] = pd.to_datetime(df['start_date'])
            df['end_date'] = pd.to_datetime(df['end_date'])
            df['year'] = df['start_date'].dt.year
            df['cause'] = df['class']
        self.reference = self.cdi.copy()
        ddataset = {'cdi': self.cdi, 'forms': self.forms}
        doa = {'cdi': 1.0, 'forms': 1.0}
        dsbuffer = {'cdi': 100, 'forms': 100}
        dtypes = {'cdi': 'polygon', 'forms': 'polygon'}
        dclass_score = {
            'cdi': {'drought': {'drought-dieback': 1.0}},
            'forms': {'clear-cut': {'anthropogenic': 1.0}},
            'reference': {'None': {'drought-dieback': 0, 'anthropogenic': 0}}
        }
        profile = {'spatial': ('step', {'start': 0, 'end': 0}),
                   'temporal': ('step', {'start': 0, 'end': 0})}
        ddataset_profile = {k: profile for k in ['cdi', 'forms', 'reference']}
        ddisturbance_profile = {
            'drought-dieback': profile,
            'anthropogenic': profile,
        }
        self.attr = Attribution(
            ddataset,
            self.reference,
            doa,
            1,
            dsbuffer,
            dtypes,
            dclass_score,
            ddataset_profile,
            ddisturbance_profile,
            start_year=2017,
            end_year=2022,
        )

    def test_dataset_creation(self):
        self.assertEqual(len(self.attr.dataset), 10)
        for col in ['start_date', 'end_date', 'class', 'dataset', 'year', 'centroid_date']:
            self.assertIn(col, self.attr.dataset.columns)
        self.assertIn('cdi', self.attr.dataset['dataset'].unique())
        self.assertIn('forms', self.attr.dataset['dataset'].unique())

    def test_similarity_matrix(self):
        sim, matrices = compute_similarity_matrix(
            self.attr.dataset,
            self.attr.dtypes_,
            {},
            self.attr.final_weighting_dict,
            weights=[1, 1]
        )
        n = len(self.attr.dataset)
        self.assertEqual(sim.shape, (n, n))
        self.assertIn('spatial', matrices)
        self.assertIn('temporal', matrices)

if __name__ == '__main__':
    unittest.main()
