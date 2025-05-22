import tempfile
import unittest
from pathlib import Path
import sys

import geopandas as gpd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from preprocessing.preprocess_cdi import preprocess_cdi

EXCERPT = Path("excerpts/excerpt_simplified_CDI_2012_2023_EPSG2154_FR.parquet")


class TestPreprocessCDI(unittest.TestCase):
    def test_preprocess_cdi(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cdi.parquet"
            gdf = preprocess_cdi(EXCERPT, out)
            self.assertTrue(out.exists())
            self.assertEqual(gdf.crs.to_epsg(), 2154)
            required = ["geometry", "start_date", "end_date", "class", "dataset", "year"]
            for col in required:
                self.assertIn(col, gdf.columns)


if __name__ == "__main__":
    unittest.main()

