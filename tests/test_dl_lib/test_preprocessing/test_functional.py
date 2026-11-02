import unittest

import numpy as np

from dl_lib.preprocessing import functional


class TestMinMaxNormalize(unittest.TestCase):
    def test_when_array_has_a_range_then_rescales_to_zero_one(self):
        # Arrange
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        # Act
        actual = functional.min_max_normalize(values)

        # Assert
        np.testing.assert_allclose(actual, expected)

    def test_when_array_is_constant_then_returns_zeros(self):
        # Arrange
        values = np.array([7.0, 7.0, 7.0])
        expected = np.array([0.0, 0.0, 0.0])

        # Act
        actual = functional.min_max_normalize(values)

        # Assert
        np.testing.assert_allclose(actual, expected)


class TestStandardize(unittest.TestCase):
    def test_when_array_has_nonzero_mean_then_centers_and_scales_it(self):
        # Arrange
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_mean = 0.0
        expected_std = 1.0

        # Act
        actual = functional.standardize(values)

        # Assert
        self.assertAlmostEqual(actual.mean(), expected_mean)
        self.assertAlmostEqual(actual.std(), expected_std)

    def test_when_array_is_constant_then_returns_zeros(self):
        # Arrange
        values = np.array([7.0, 7.0, 7.0])
        expected = np.array([0.0, 0.0, 0.0])

        # Act
        actual = functional.standardize(values)

        # Assert
        np.testing.assert_allclose(actual, expected)
