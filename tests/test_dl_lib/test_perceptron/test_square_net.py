# import unittest
# from functools import partial
#
# import numpy as np
#
# from dl_lib.losses import functional as losses_functional
# from dl_lib.perceptron import functional
# from dl_lib.perceptron.square_net import SquareNet
#
# SQUARE_DATASET = [(np.array([float(x)]), float(x**2)) for x in range(-3, 4)]
#
#
# def _loss_given(
#     model: SquareNet,
#     dataset: list[tuple[np.ndarray, float]],
#     targets: np.ndarray,
#     param_name: str,
#     param: np.ndarray | float,
# ) -> float:
#     original = getattr(model, param_name)
#     setattr(model, param_name, param)
#     predictions = np.array([model.forward(x) for x, _ in dataset])
#     setattr(model, param_name, original)
#     return losses_functional.mse(targets, predictions)
#
#
# def _loss_given_output_bias(
#     model: SquareNet,
#     dataset: list[tuple[np.ndarray, float]],
#     targets: np.ndarray,
#     bias: np.ndarray,
# ) -> float:
#     return _loss_given(model, dataset, targets, "output_bias", float(bias[0]))
#
#
# def _current_loss(
#     model: SquareNet, dataset: list[tuple[np.ndarray, float]], targets: np.ndarray
# ) -> float:
#     predictions = np.array([model.forward(x) for x, _ in dataset])
#     return losses_functional.mse(targets, predictions)
#
#
# class TestInit(unittest.TestCase):
#     def test_when_called_then_builds_a_hidden_layer_and_a_linear_output_unit(self):
#         # Arrange
#         hidden_units = 4
#
#         # Act
#         model = SquareNet(hidden_units=hidden_units, seed=42)
#
#         # Assert
#         self.assertEqual(model.hidden_weights.shape, (hidden_units,))
#         self.assertEqual(model.hidden_bias.shape, (hidden_units,))
#         self.assertEqual(model.output_weights.shape, (hidden_units,))
#         self.assertIsInstance(model.output_bias, float)
#
#
# class TestForward(unittest.TestCase):
#     def test_when_called_then_output_is_finite(self):
#         # Arrange
#         model = SquareNet(seed=42)
#         x = np.array([2.0])
#
#         # Act
#         actual = model.forward(x)
#
#         # Assert
#         self.assertTrue(np.isfinite(actual))
#
#
# class TestTraining(unittest.TestCase):
#     def test_when_a_single_gradient_descent_step_is_applied_to_the_hidden_weights_then_loss_decreases(
#         self,
#     ):
#         # Arrange
#         model = SquareNet(hidden_units=4, low=-1.0, high=1.0, seed=42)
#         targets = np.array([y for _, y in SQUARE_DATASET])
#         learning_rate = 0.01
#         hidden_weights_loss = partial(
#             _loss_given, model, SQUARE_DATASET, targets, "hidden_weights"
#         )
#
#         initial_loss = _current_loss(model, SQUARE_DATASET, targets)
#
#         # Act
#         gradient = functional.numerical_gradient(
#             hidden_weights_loss, model.hidden_weights
#         )
#         model.hidden_weights = functional.gradient_descent_step(
#             model.hidden_weights, gradient, learning_rate
#         )
#         final_loss = _current_loss(model, SQUARE_DATASET, targets)
#
#         # Assert
#         self.assertLess(final_loss, initial_loss)
#
#     def test_when_trained_for_a_few_steps_then_loss_decreases(self):
#         # Arrange
#         model = SquareNet(hidden_units=4, low=-1.0, high=1.0, seed=42)
#         targets = np.array([y for _, y in SQUARE_DATASET])
#         learning_rate = 0.01
#         epochs = 10
#
#         initial_loss = _current_loss(model, SQUARE_DATASET, targets)
#
#         # Act
#         for _ in range(epochs):
#             for param_name in ("hidden_weights", "hidden_bias", "output_weights"):
#                 param = getattr(model, param_name)
#                 gradient = functional.numerical_gradient(
#                     partial(_loss_given, model, SQUARE_DATASET, targets, param_name),
#                     param,
#                 )
#                 setattr(
#                     model,
#                     param_name,
#                     functional.gradient_descent_step(param, gradient, learning_rate),
#                 )
#             bias_array = np.array([model.output_bias])
#             gradient = functional.numerical_gradient(
#                 partial(_loss_given_output_bias, model, SQUARE_DATASET, targets),
#                 bias_array,
#             )
#             model.output_bias = functional.gradient_descent_step(
#                 bias_array, gradient, learning_rate
#             )[0]
#
#         final_loss = _current_loss(model, SQUARE_DATASET, targets)
#
#         # Assert
#         self.assertLess(final_loss, initial_loss)
