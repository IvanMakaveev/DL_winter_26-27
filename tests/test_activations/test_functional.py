# import unittest
#
# import numpy as np
# import torch
#
# from dl_lib.activations import functional
#
#
# class TestSigmoid(unittest.TestCase):
#
#     def test_when_input_is_zero_then_returns_one_half(self):
#         # Arrange
#         x = np.array([0.0])
#         expected = 0.5
#
#         # Act
#         actual = functional.sigmoid(x)[0]
#
#         # Assert
#         self.assertAlmostEqual(actual, expected)
#
#     def test_when_input_is_very_negative_then_approaches_zero(self):
#         # Arrange
#         x = np.array([-100.0])
#
#         # Act
#         actual = functional.sigmoid(x)[0]
#
#         # Assert
#         self.assertAlmostEqual(actual, 0.0)
#
#     def test_when_input_is_very_positive_then_approaches_one(self):
#         # Arrange
#         x = np.array([100.0])
#
#         # Act
#         actual = functional.sigmoid(x)[0]
#
#         # Assert
#         self.assertAlmostEqual(actual, 1.0)
#
#     def test_when_called_then_matches_torch_sigmoid(self):
#         # Arrange
#         x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
#         expected = torch.sigmoid(torch.from_numpy(x)).numpy()
#
#         # Act
#         actual = functional.sigmoid(x)
#
#         # Assert
#         np.testing.assert_allclose(actual, expected)
#
#
# class TestRelu(unittest.TestCase):
#
#     def test_when_input_is_negative_then_returns_zero(self):
#         # Arrange
#         x = np.array([-5.0, -1.0])
#         expected = np.array([0.0, 0.0])
#
#         # Act
#         actual = functional.relu(x)
#
#         # Assert
#         np.testing.assert_allclose(actual, expected)
#
#     def test_when_input_is_positive_then_returns_input_unchanged(self):
#         # Arrange
#         x = np.array([2.0, 7.5])
#
#         # Act
#         actual = functional.relu(x)
#
#         # Assert
#         np.testing.assert_allclose(actual, x)
#
#     def test_when_called_then_matches_torch_relu(self):
#         # Arrange
#         x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
#         expected = torch.nn.functional.relu(torch.from_numpy(x)).numpy()
#
#         # Act
#         actual = functional.relu(x)
#
#         # Assert
#         np.testing.assert_allclose(actual, expected)
#
#
# class TestTanh(unittest.TestCase):
#
#     def test_when_input_is_zero_then_returns_zero(self):
#         # Arrange
#         x = np.array([0.0])
#         expected = 0.0
#
#         # Act
#         actual = functional.tanh(x)[0]
#
#         # Assert
#         self.assertAlmostEqual(actual, expected)
#
#     def test_when_input_is_very_negative_then_approaches_negative_one(self):
#         # Arrange
#         x = np.array([-100.0])
#
#         # Act
#         actual = functional.tanh(x)[0]
#
#         # Assert
#         self.assertAlmostEqual(actual, -1.0)
#
#     def test_when_called_then_matches_torch_tanh(self):
#         # Arrange
#         x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
#         expected = torch.tanh(torch.from_numpy(x)).numpy()
#
#         # Act
#         actual = functional.tanh(x)
#
#         # Assert
#         np.testing.assert_allclose(actual, expected)
#
#
# class TestSoftmax(unittest.TestCase):
#
#     def test_when_called_then_probabilities_sum_to_one(self):
#         # Arrange
#         x = np.array([1.0, 2.0, 3.0])
#         expected = 1.0
#
#         # Act
#         actual = functional.softmax(x).sum()
#
#         # Assert
#         self.assertAlmostEqual(actual, expected)
#
#     def test_when_input_has_large_values_then_does_not_overflow(self):
#         # Arrange
#         x = np.array([1000.0, 1001.0, 1002.0])
#
#         # Act
#         actual = functional.softmax(x)
#
#         # Assert
#         self.assertTrue(np.all(np.isfinite(actual)))
#         self.assertAlmostEqual(actual.sum(), 1.0)
#
#     def test_when_called_then_matches_torch_softmax(self):
#         # Arrange
#         x = np.array([1.0, 2.0, 3.0])
#         expected = torch.softmax(torch.from_numpy(x), dim=-1).numpy()
#
#         # Act
#         actual = functional.softmax(x)
#
#         # Assert
#         np.testing.assert_allclose(actual, expected)
#
#     def test_when_input_is_batched_then_normalizes_each_row_independently(self):
#         # Arrange
#         x = np.array([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]])
#         expected_row_sums = np.array([1.0, 1.0])
#
#         # Act
#         actual = functional.softmax(x)
#
#         # Assert
#         np.testing.assert_allclose(actual.sum(axis=-1), expected_row_sums)
