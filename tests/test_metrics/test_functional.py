# import unittest
#
# import numpy as np
#
# from dl_lib.metrics import functional
#
#
# class TestApplyThreshold(unittest.TestCase):
#
#     def test_when_probabilities_straddle_threshold_then_splits_them(self):
#         # Arrange
#         probabilities = np.array([0.1, 0.8, 0.4, 0.7, 0.9])
#         threshold = 0.5
#         expected = np.array([0, 1, 0, 1, 1])
#
#         # Act
#         actual = functional.apply_threshold(probabilities, threshold)
#
#         # Assert
#         np.testing.assert_array_equal(actual, expected)
#
#     def test_when_threshold_rises_then_fewer_predictions_are_positive(self):
#         # Arrange
#         probabilities = np.array([0.1, 0.8, 0.4, 0.7, 0.9])
#
#         # Act
#         low_threshold_positives = functional.apply_threshold(probabilities, 0.1).sum()
#         high_threshold_positives = functional.apply_threshold(probabilities, 0.9).sum()
#
#         # Assert
#         self.assertGreater(low_threshold_positives, high_threshold_positives)
#
#
# class TestConfusionMatrix(unittest.TestCase):
#
#     def test_when_called_then_counts_each_quadrant(self):
#         # Arrange
#         y_true = np.array([1, 0, 1, 0, 1])
#         y_pred = np.array([1, 0, 0, 1, 1])
#         expected = {'tp': 2, 'tn': 1, 'fp': 1, 'fn': 1}
#
#         # Act
#         actual = functional.confusion_matrix(y_true, y_pred)
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestAccuracy(unittest.TestCase):
#
#     def test_when_called_then_matches_hand_calculation(self):
#         # Arrange
#         y_true = np.array([1, 0, 1, 0, 1])
#         y_pred = np.array([1, 0, 0, 1, 1])
#         expected = 3 / 5
#
#         # Act
#         actual = functional.accuracy(y_true, y_pred)
#
#         # Assert
#         self.assertAlmostEqual(actual, expected)
#
#     def test_when_dataset_is_imbalanced_and_all_predicted_negative_then_accuracy_is_still_high(
#             self):
#         # Arrange
#         y_true = np.array([0] * 990 + [1] * 10)
#         y_pred = np.zeros(1000, dtype=int)
#         expected = 0.99
#
#         # Act
#         actual = functional.accuracy(y_true, y_pred)
#
#         # Assert
#         self.assertAlmostEqual(actual, expected)
#
#
# class TestPrecision(unittest.TestCase):
#
#     def test_when_called_then_matches_hand_calculation(self):
#         # Arrange
#         y_true = np.array([1, 0, 1, 0, 1])
#         y_pred = np.array([1, 0, 0, 1, 1])
#         expected = 2 / 3
#
#         # Act
#         actual = functional.precision(y_true, y_pred)
#
#         # Assert
#         self.assertAlmostEqual(actual, expected)
#
#     def test_when_no_positive_predictions_then_returns_zero(self):
#         # Arrange
#         y_true = np.array([1, 0, 1])
#         y_pred = np.array([0, 0, 0])
#         expected = 0.0
#
#         # Act
#         actual = functional.precision(y_true, y_pred)
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestRecall(unittest.TestCase):
#
#     def test_when_called_then_matches_hand_calculation(self):
#         # Arrange
#         y_true = np.array([1, 0, 1, 0, 1])
#         y_pred = np.array([1, 0, 0, 1, 1])
#         expected = 2 / 3
#
#         # Act
#         actual = functional.recall(y_true, y_pred)
#
#         # Assert
#         self.assertAlmostEqual(actual, expected)
#
#     def test_when_no_actual_positives_then_returns_zero(self):
#         # Arrange
#         y_true = np.array([0, 0, 0])
#         y_pred = np.array([1, 0, 1])
#         expected = 0.0
#
#         # Act
#         actual = functional.recall(y_true, y_pred)
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestF1Score(unittest.TestCase):
#
#     def test_when_called_then_matches_harmonic_mean_of_precision_and_recall(self):
#         # Arrange
#         y_true = np.array([1, 0, 1, 0, 1])
#         y_pred = np.array([1, 0, 0, 1, 1])
#         p = functional.precision(y_true, y_pred)
#         r = functional.recall(y_true, y_pred)
#         expected = 2 * p * r / (p + r)
#
#         # Act
#         actual = functional.f1_score(y_true, y_pred)
#
#         # Assert
#         self.assertAlmostEqual(actual, expected)
#
#     def test_when_precision_and_recall_are_both_zero_then_returns_zero(self):
#         # Arrange
#         y_true = np.array([0, 0, 0])
#         y_pred = np.array([0, 0, 0])
#         expected = 0.0
#
#         # Act
#         actual = functional.f1_score(y_true, y_pred)
#
#         # Assert
#         self.assertEqual(actual, expected)
