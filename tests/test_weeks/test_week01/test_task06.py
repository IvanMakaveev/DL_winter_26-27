import unittest

import numpy as np
import pytest

from weeks.week01 import task06


class TestPrintArrayData(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_vector_is_given_then_prints_vector_data(self):
        # Arrange
        array = np.array([1, 2, 3])

        # Act
        task06.print_array_data(array)
        actual_log = self.capsys.readouterr().out

        # Assert
        self.assertIn("1D array (vector): shape=(3,), dtype=int", actual_log)
        self.assertIn("ndim=1, size=3", actual_log)
        self.assertIn("[1 2 3]", actual_log)

    def test_when_matrix_is_given_then_prints_matrix_data(self):
        # Arrange
        array = np.array([[1, 2], [3, 4]])

        # Act
        task06.print_array_data(array)
        actual_log = self.capsys.readouterr().out

        # Assert
        self.assertIn("2D array (matrix): shape=(2, 2), dtype=int", actual_log)
        self.assertIn("ndim=2, size=4", actual_log)
        self.assertIn("[[1 2]", actual_log)
        self.assertIn("[3 4]]", actual_log)

    def test_when_volume_is_given_then_prints_volume_data(self):
        # Arrange
        array = np.zeros((2, 2, 2), dtype=int)

        # Act
        task06.print_array_data(array)
        actual_log = self.capsys.readouterr().out

        # Assert
        self.assertIn("3D array (volume): shape=(2, 2, 2), dtype=int", actual_log)
        self.assertIn("ndim=3, size=8", actual_log)

    def test_when_hyper_volume_is_given_then_prints_hyper_volume_data(self):
        # Arrange
        array = np.zeros((2, 2, 2, 2), dtype=int)

        # Act
        task06.print_array_data(array)
        actual_log = self.capsys.readouterr().out

        # Assert
        self.assertIn(
            "4D array (hyper-volume): shape=(2, 2, 2, 2), dtype=int", actual_log
        )
        self.assertIn("ndim=4, size=16", actual_log)


class TestMain(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def inject_fixtures(self, capsys):
        self.capsys = capsys

    def test_when_called_then_prints_all_example_arrays(self):
        # Arrange
        expected_logs = (
            "1D array (vector): shape=(5,), dtype=int64, ndim=1, size=5",
            "[1 2 3 4 5]",
            "2D array (matrix): shape=(2, 3), dtype=int64, ndim=2, size=6",
            "[[1 2 3]",
            "[4 5 6]]",
            "3D array (volume): shape=(2, 3, 4), dtype=int64, ndim=3, size=24",
            "[[[ 0  1  2  3]",
            "[[12 13 14 15]",
        )

        # Act
        task06.main()
        actual_log = self.capsys.readouterr().out

        # Assert
        for expected_log in expected_logs:
            self.assertIn(expected_log, actual_log)
