import tempfile
import unittest
from pathlib import Path
import sys

import geopandas as gpd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from preprocessing.preprocess_forms import preprocess_forms

EXCERPT = Path("excerpts/excerpt_simplified_FORMS_clearcut_2017_2020_EPSG2154.parquet")


class TestPreprocessForms(unittest.TestCase):
    def test_preprocess_forms(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "forms.parquet"
            gdf = preprocess_forms(EXCERPT, out)
            self.assertTrue(out.exists())
            self.assertEqual(gdf.crs.to_epsg(), 2154)
            required = ["geometry", "start_date", "end_date", "class", "dataset", "year"]
            for col in required:
                self.assertIn(col, gdf.columns)


if __name__ == "__main__":
    unittest.main()

