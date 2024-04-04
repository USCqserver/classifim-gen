import unittest
import numpy as np
import classifim_bench.metric

class TestStraightLineDistance(unittest.TestCase):
    def _test_2D(self, cls):
        grid = [np.arange(4) / 4, np.arange(3) / 4]
        g_up = np.array([[0, 0], [0, 1]])
        g_right = np.array([[1, 0], [0, 0]])
        g_ur = np.array([[1, 1], [1, 1]])
        g_ul = np.array([[1, -1], [-1, 1]])
        metric = np.array([
            [16 * g_up, 1 * g_up, 1 * g_ur],
            [1 * g_right, 0 * g_up, 0 * g_up],
            [3 * g_right, 0 * g_up, 1 * g_ul],
            [5 * g_right, 6 * g_right, 7 * g_right]])
        distance_calc = cls(grid, metric)
        self.assertEqual(distance_calc([0, 0], [0, 0]), 0)
        self.assertEqual(distance_calc([0, 0], [0, 3/8]), 3/4)
        self.assertAlmostEqual(
            distance_calc([1/8, 0], [1/8, 3/8]),
            3 / 4 / np.sqrt(2), places=15)
        self.assertEqual(distance_calc([0, 0], [3/8, 0]), 1/4)
        self.assertAlmostEqual(
            distance_calc([0, 1/8], [3/8, 1/8]),
            1 / 4 / np.sqrt(2), places=15)
        self.assertEqual(distance_calc([-1/8, 3/8], [1/8, 5/8]), 1/2)
        self.assertAlmostEqual(
            distance_calc([-1/8, -1/8], [7/8, 5/8]),
            (3 + 1/3 + 1/12 + np.sqrt(7)) / 4, places=15)

    def test_2D_Py(self):
        self._test_2D(classifim_bench.metric.StraightLineDistance)

    def test_2D_Cpp(self):
        self._test_2D(classifim_bench.metric.StraightLineDistanceCpp)

    def _test_1D(self, cls):
        grid = [np.arange(4)]
        metric = np.array([1, 2, 3, 4]).reshape((4, 1, 1))
        distance_calc = cls(grid, metric)
        self.assertEqual(distance_calc([0], [0]), 0)
        self.assertAlmostEqual(
            distance_calc([0], [3]),
            0.5 + np.sqrt(2) + np.sqrt(3) + 1.0, places=15)

    def test_1D_Py(self):
        self._test_1D(classifim_bench.metric.StraightLineDistance)

    def test_1D_Cpp(self):
        self._test_1D(classifim_bench.metric.StraightLineDistanceCpp)

    def _test_2D_flat(self, cls):
        grid = [(0.5 + np.arange(4)) / 4, (0.5 + np.arange(4)) / 4]
        gflat = np.eye(2)
        metric = np.array([[gflat] * 4] * 4)
        distance_calc = cls(grid, metric)
        rng = np.random.default_rng(1)
        for _ in range(20):
            a = rng.random(2)
            b = rng.random(2)
            self.assertAlmostEqual(
                distance_calc(a, b),
                np.linalg.norm(a - b), places=15)

    def test_2D_flat_Py(self):
        self._test_2D_flat(classifim_bench.metric.StraightLineDistance)

    def test_2D_flat_Cpp(self):
        self._test_2D_flat(classifim_bench.metric.StraightLineDistanceCpp)

class TestCountInversions(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(classifim_bench.metric.count_inversions([]), 0)

    def test_simple(self):
        self.assertEqual(classifim_bench.metric.count_inversions([0, 1, 2, 3]), 0)
        self.assertEqual(classifim_bench.metric.count_inversions([3, 0, 1, 2]), 3)
        self.assertEqual(classifim_bench.metric.count_inversions([3, 2, 1, 0]), 6)

class TestComputeRankingError(unittest.TestCase):
    def test_simple(self):
        compute_ranking_error = classifim_bench.metric.compute_ranking_error
        a0 = np.array([0, 1, 2, 3])
        ii = np.array([3, 1, 0, 2]) # Random permutation of [0, 1, 2, 3]
        self.assertEqual(compute_ranking_error(a0[ii], a0[ii]), 0.0)
        a1 = np.array([3, 0, 1, 2])
        self.assertEqual(compute_ranking_error(a0[ii], a1[ii]), 0.5)
        self.assertEqual(compute_ranking_error(a1[ii], a0[ii]), 0.5)
        a1 = np.array([3, 2, 1, 0])
        self.assertEqual(compute_ranking_error(a0[ii], a1[ii]), 1.0)
        self.assertEqual(compute_ranking_error(a1[ii], a0[ii]), 1.0)

    def test_ties(self):
        compute_ranking_error = classifim_bench.metric.compute_ranking_error
        a0 = np.array([.0, .1, .1, .2])
        ii = np.array([2, 0, 3, 1]) # Random permutation of [0, 1, 2, 3]
        self.assertEqual(compute_ranking_error(a0[ii], a0[ii]), 1 / 12)
        a1 = np.array([3, 0, 1, 2])
        self.assertEqual(compute_ranking_error(a0[ii], a1[ii]), 7 / 12)
        self.assertEqual(compute_ranking_error(a1[ii], a0[ii]), 7 / 12)
        a1 = np.array([3, 2, 1, 0])
        self.assertEqual(compute_ranking_error(a0[ii], a1[ii]), 11 / 12)
        self.assertEqual(compute_ranking_error(a1[ii], a0[ii]), 11 / 12)
