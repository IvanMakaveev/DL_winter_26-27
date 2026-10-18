# import unittest
#
# import numpy as np
# import torch
#
# from dl_lib.losses import functional
#
#
# class TestMae(unittest.TestCase):
#
#     def test_when_called_then_matches_hand_calculation(self):
#         # Arrange
#         y_true = np.array([3.0, 5.0, 7.0, 9.0])
#         y_pred = np.array([2.0, 6.0, 8.0, 7.0])
#         expected = 1.25
#
#         # Act
#         actual = functional.mae(y_true, y_pred)
#
#         # Assert
#         self.assertAlmostEqual(actual, expected)
#
#     def test_when_called_then_matches_torch_l1_loss(self):
#         # Arrange
#         y_true = np.array([3.0, 5.0, 7.0, 9.0])
#         y_pred = np.array([2.0, 6.0, 8.0, 7.0])
#         expected = torch.nn.functional.l1_loss(torch.from_numpy(y_pred),
#                                                torch.from_numpy(y_true)).item()
#
#         # Act
#         actual = functional.mae(y_true, y_pred)
#
#         # Assert
#         self.assertAlmostEqual(actual, expected)
#
#
# class TestMse(unittest.TestCase):
#
#     def test_when_called_then_matches_hand_calculation(self):
#         # Arrange
#         y_true = np.array([3.0, 5.0, 7.0, 9.0])
#         y_pred = np.array([2.0, 6.0, 8.0, 7.0])
#         expected = 1.75
#
#         # Act
#         actual = functional.mse(y_true, y_pred)
#
#         # Assert
#         self.assertAlmostEqual(actual, expected)
#
#     def test_when_called_then_matches_torch_mse_loss(self):
#         # Arrange
#         y_true = np.array([3.0, 5.0, 7.0, 9.0])
#         y_pred = np.array([2.0, 6.0, 8.0, 7.0])
#         expected = torch.nn.functional.mse_loss(torch.from_numpy(y_pred),
#                                                 torch.from_numpy(y_true)).item()
#
#         # Act
#         actual = functional.mse(y_true, y_pred)
#
#         # Assert
#         self.assertAlmostEqual(actual, expected)
#
#     def test_when_one_prediction_is_far_off_then_penalizes_it_more_than_mae_does(self):
#         # Arrange
#         y_true = np.array([1.0, 1.0, 1.0, 1.0])
#         y_pred_close = np.array([2.0, 1.0, 1.0, 1.0])
#         y_pred_far = np.array([10.0, 1.0, 1.0, 1.0])
#
#         # Act
#         mae_ratio = functional.mae(y_true, y_pred_far) / functional.mae(y_true, y_pred_close)
#         mse_ratio = functional.mse(y_true, y_pred_far) / functional.mse(y_true, y_pred_close)
#
#         # Assert
#         self.assertGreater(mse_ratio, mae_ratio)
#
#
# class TestRmse(unittest.TestCase):
#
#     def test_when_called_then_matches_hand_calculation(self):
#         # Arrange
#         y_true = np.array([3.0, 5.0, 7.0, 9.0])
#         y_pred = np.array([2.0, 6.0, 8.0, 7.0])
#         expected = 1.3229
#
#         # Act
#         actual = functional.rmse(y_true, y_pred)
#
#         # Assert
#         self.assertAlmostEqual(actual, expected, places=4)
#
#     def test_when_called_then_equals_square_root_of_mse(self):
#         # Arrange
#         y_true = np.array([3.0, 5.0, 7.0, 9.0])
#         y_pred = np.array([2.0, 6.0, 8.0, 7.0])
#         expected = np.sqrt(functional.mse(y_true, y_pred))
#
#         # Act
#         actual = functional.rmse(y_true, y_pred)
#
#         # Assert
#         self.assertAlmostEqual(actual, expected)
#
#
# class TestBinaryCrossEntropy(unittest.TestCase):
#
#     def test_when_called_then_matches_hand_calculation(self):
#         # Arrange
#         y_true = np.array([1.0, 0.0])
#         p = np.array([0.9, 0.1])
#         expected = -np.log(0.9)
#
#         # Act
#         actual = functional.binary_cross_entropy(y_true, p)
#
#         # Assert
#         self.assertAlmostEqual(actual, expected)
#
#     def test_when_called_then_matches_torch_binary_cross_entropy(self):
#         # Arrange
#         y_true = np.array([1.0, 0.0, 1.0])
#         p = np.array([0.9, 0.2, 0.6])
#         expected = torch.nn.functional.binary_cross_entropy(torch.from_numpy(p),
#                                                             torch.from_numpy(y_true)).item()
#
#         # Act
#         actual = functional.binary_cross_entropy(y_true, p)
#
#         # Assert
#         self.assertAlmostEqual(actual, expected)
#
#     def test_when_confidently_wrong_then_penalizes_more_than_confidently_right(self):
#         # Arrange
#         y_true = np.array([1.0])
#
#         # Act
#         confidently_right = functional.binary_cross_entropy(y_true, np.array([0.999]))
#         confidently_wrong = functional.binary_cross_entropy(y_true, np.array([0.001]))
#
#         # Assert
#         self.assertGreater(confidently_wrong, confidently_right)
#
#
# class TestCategoricalCrossEntropy(unittest.TestCase):
#
#     def test_when_called_then_matches_hand_calculation(self):
#         # Arrange
#         y_true = np.array([0, 1, 2])
#         probabilities = np.array([
#             [0.7, 0.2, 0.1],
#             [0.1, 0.8, 0.1],
#             [0.2, 0.2, 0.6],
#         ])
#         expected = -np.mean(np.log([0.7, 0.8, 0.6]))
#
#         # Act
#         actual = functional.categorical_cross_entropy(y_true, probabilities)
#
#         # Assert
#         self.assertAlmostEqual(actual, expected)
#
#     def test_when_called_then_matches_torch_nll_loss_on_log_probabilities(self):
#         # Arrange
#         y_true = np.array([0, 1, 2])
#         probabilities = np.array([
#             [0.7, 0.2, 0.1],
#             [0.1, 0.8, 0.1],
#             [0.2, 0.2, 0.6],
#         ])
#         expected = torch.nn.functional.nll_loss(torch.log(torch.from_numpy(probabilities)),
#                                                 torch.from_numpy(y_true)).item()
#
#         # Act
#         actual = functional.categorical_cross_entropy(y_true, probabilities)
#
#         # Assert
#         self.assertAlmostEqual(actual, expected)
#
#     def test_when_confidently_wrong_then_penalizes_more_than_confidently_right(self):
#         # Arrange
#         y_true = np.array([0])
#
#         # Act
#         confidently_right = functional.categorical_cross_entropy(
#             y_true, np.array([[0.999, 0.0005, 0.0005]]))
#         confidently_wrong = functional.categorical_cross_entropy(
#             y_true, np.array([[0.001, 0.4995, 0.4995]]))
#
#         # Assert
#         self.assertGreater(confidently_wrong, confidently_right)
