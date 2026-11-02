# import unittest
#
# from dl_lib.activations import functional as activations_functional
# from dl_lib.losses import functional as losses_functional
# from dl_lib.perceptron import datasets, training
# from dl_lib.perceptron.perceptron import Perceptron
#
#
# class TestTrain(unittest.TestCase):
#     def test_when_called_then_loss_decreases(self):
#         # Arrange
#         model = Perceptron(2, low=-1.0, high=1.0, use_bias=True, seed=42)
#         dataset = datasets.and_dataset()
#         learning_rate = 0.1
#         epochs = 50
#
#         # Act
#         history = training.train(model, dataset, learning_rate, epochs)
#
#         # Assert
#         initial_loss = history[0][1]
#         final_loss = history[-1][1]
#         self.assertLess(final_loss, initial_loss)
#
#     def test_when_bce_loss_given_then_loss_decreases(self):
#         # Arrange
#         model = Perceptron(
#             2,
#             low=-1.0,
#             high=1.0,
#             use_bias=True,
#             activation=activations_functional.sigmoid,
#             seed=42,
#         )
#         dataset = datasets.and_dataset()
#         learning_rate = 0.1
#         epochs = 50
#
#         # Act
#         history = training.train(
#             model,
#             dataset,
#             learning_rate,
#             epochs,
#             loss=losses_functional.binary_cross_entropy,
#         )
#
#         # Assert
#         initial_loss = history[0][1]
#         final_loss = history[-1][1]
#         self.assertLess(final_loss, initial_loss)
#
#     def test_when_log_every_is_given_then_history_has_one_entry_per_logged_epoch(self):
#         # Arrange
#         model = Perceptron(2, use_bias=True, seed=42)
#         dataset = datasets.and_dataset()
#         epochs = 10
#         log_every = 2
#         expected_logged_epochs = [2, 4, 6, 8, 10]
#
#         # Act
#         history = training.train(
#             model, dataset, learning_rate=0.1, epochs=epochs, log_every=log_every
#         )
#
#         # Assert
#         actual_logged_epochs = [epoch for epoch, _ in history]
#         self.assertEqual(actual_logged_epochs, expected_logged_epochs)
#
#     def test_when_model_has_no_bias_then_bias_stays_zero(self):
#         # Arrange
#         model = Perceptron(2, use_bias=False, seed=42)
#         dataset = datasets.and_dataset()
#         expected_bias = 0.0
#
#         # Act
#         training.train(model, dataset, learning_rate=0.1, epochs=10)
#
#         # Assert
#         self.assertEqual(model.bias, expected_bias)
