import tempfile
import unittest
from pathlib import Path
import sys

import geopandas as gpd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from preprocessing.preprocess_firepolygons import preprocess_firepolygons

EXCERPT = Path("excerpts/excerpt_simplified_firepolygons_2017_2020_FR_EPSG2154.parquet")


class TestPreprocessFirepolygons(unittest.TestCase):
    def test_preprocess_firepolygons(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "fire.parquet"
            gdf = preprocess_firepolygons(EXCERPT, out)
            self.assertTrue(out.exists())
            self.assertEqual(gdf.crs.to_epsg(), 2154)
            required = ["geometry", "start_date", "end_date", "class", "dataset", "year"]
            for col in required:
                self.assertIn(col, gdf.columns)


if __name__ == "__main__":
    unittest.main()

