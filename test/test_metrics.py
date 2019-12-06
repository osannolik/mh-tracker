import unittest

import numpy as np

from mht.metrics import (MOTMetric)

class MOTMetricsTest(unittest.TestCase):

    def setUp(self):
        ground_truth = [
            {'a': np.array([1.0, 2.0]), 'b': np.array([2.0, 2.0]), 'c': np.array([3.0, 2.0])},
            {'a': np.array([2.0, 2.0]), 'b': np.array([3.0, 2.0]), 'c': np.array([4.0, 2.0])}
        ]

        tracks = [
            {1: np.array([0.0, 0.0]), 2: np.array([1.0, 1.0])},
            {1: np.array([2.0, 1.0]), 2: np.array([3.0, 2.0]), 3: np.array([4.0, 3.0])}
        ]

        self.metric = MOTMetric(ground_truth, tracks, 2.0)

    def test_mota(self):
        mota = self.metric.MOTA()
        self.assertEqual(mota, 0.5)

    def test_motp(self):
        motp = self.metric.MOTP()
        self.assertEqual(motp, 1.25)

if __name__ == '__main__':
    unittest.main()
