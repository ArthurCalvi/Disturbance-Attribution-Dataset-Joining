import tempfile
import unittest
from pathlib import Path
import sys

import geopandas as gpd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from preprocessing.preprocess_hm import preprocess_hm

EXCERPT = Path("excerpts/excerpt_simplified_health-monitoring_2007-2023_EPSG2154_FR.parquet")


class TestPreprocessHM(unittest.TestCase):
    def test_preprocess_hm(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "hm.parquet"
            gdf = preprocess_hm(EXCERPT, out)
            self.assertTrue(out.exists())
            self.assertEqual(gdf.crs.to_epsg(), 2154)
            required = ["geometry", "start_date", "end_date", "class", "dataset", "year"]
            for col in required:
                self.assertIn(col, gdf.columns)


if __name__ == "__main__":
    unittest.main()

