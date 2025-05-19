import unittest

from src.join_datasets.helpers import modify_dsbuffer, weighting_function


class TestHelpers(unittest.TestCase):
    def test_modify_dsbuffer(self):
        dsbuffer = {"a": 100, "b": None}
        res = modify_dsbuffer(dsbuffer, 50)
        self.assertEqual(res["a"], 50)
        self.assertIsNone(res["b"])

    def test_weighting_function(self):
        self.assertEqual(weighting_function(0, 10, 5), 1)
        self.assertEqual(weighting_function(10, 10, 5), 1)
        val = weighting_function(20, 10, 5)
        self.assertTrue(0 < val < 1)


if __name__ == "__main__":
    unittest.main()
