import unittest

import numpy as np

import classifim_gen.peaks as peaks

class TestFindPeaks(unittest.TestCase):
    def test_flat_top(self):
        x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        y = np.array([1.0, 0.9, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.9, 1.0])
        actual = peaks.find_peaks(x, y, 1)
        expected = [0.45]
        np.testing.assert_array_equal(actual, expected)

    def test_single_peak(self):
        x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        y = np.array([1.0, 0.8, 0.81, 0.8, 0.9, 1.0, 0.9, 0.8, 0.9, 1.0])
        actual = peaks.find_peaks(x, y, 1)
        expected = [0.5]
        np.testing.assert_array_equal(actual, expected)

    def test_two_peaks(self):
        x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        y = np.array([1.0, 0.8, 0.81, 0.8, 0.9, 1.0, 0.9, 0.8, 0.9, 1.0])
        actual = peaks.find_peaks(x, y, 2)
        expected = [0.2, 0.5]
        np.testing.assert_array_equal(actual, expected)
