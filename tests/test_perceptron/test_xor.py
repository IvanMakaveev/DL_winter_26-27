# import unittest
# from functools import partial
#
# import numpy as np
#
# from dl_lib.losses import functional as losses_functional
# from dl_lib.perceptron import datasets, functional
# from dl_lib.perceptron.xor import Xor, forward
#
#
# def _loss_given(
#     model: Xor,
#     dataset: list[tuple[np.ndarray, float]],
#     targets: np.ndarray,
#     param_name: str,
#     param: np.ndarray | float,
# ) -> float:
#     original = getattr(model, param_name)
#     setattr(model, param_name, param)
#     predictions = np.array([model.forward(x) for x, _ in dataset])
#     setattr(model, param_name, original)
#     return losses_functional.binary_cross_entropy(targets, predictions)
#
#
# def _loss_given_output_bias(
#     model: Xor,
#     dataset: list[tuple[np.ndarray, float]],
#     targets: np.ndarray,
#     bias: np.ndarray,
# ) -> float:
#     return _loss_given(model, dataset, targets, "output_bias", float(bias[0]))
#
#
# def _current_loss(
#     model: Xor, dataset: list[tuple[np.ndarray, float]], targets: np.ndarray
# ) -> float:
#     predictions = np.array([model.forward(x) for x, _ in dataset])
#     return losses_functional.binary_cross_entropy(targets, predictions)
#
#
# class TestInit(unittest.TestCase):
#     def test_when_called_then_builds_two_hidden_units_and_one_output_unit(self):
#         # Arrange / Act
#         model = Xor(seed=42)
#
#         # Assert
#         self.assertEqual(model.hidden_weights.shape, (2, 2))
#         self.assertEqual(model.hidden_bias.shape, (2,))
#         self.assertEqual(model.output_weights.shape, (2,))
#         self.assertIsInstance(model.output_bias, float)
#
#
# class TestForward(unittest.TestCase):
#     def test_when_called_then_output_is_a_probability(self):
#         # Arrange
#         model = Xor(seed=42)
#         x = np.array([1.0, 0.0])
#
#         # Act
#         actual = model.forward(x)
#
#         # Assert
#         self.assertGreater(actual, 0.0)
#         self.assertLess(actual, 1.0)
#
#
# class TestModuleForward(unittest.TestCase):
#     def test_when_called_then_matches_the_models_own_forward(self):
#         # Arrange
#         model = Xor(seed=42)
#         x1, x2 = 1.0, 0.0
#         expected = model.forward(np.array([x1, x2]))
#
#         # Act
#         actual = forward(model, x1, x2)
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestTraining(unittest.TestCase):
#     def test_when_a_single_gradient_descent_step_is_applied_to_the_hidden_weights_then_loss_decreases(
#         self,
#     ):
#         # Arrange
#         model = Xor(seed=42)
#         dataset = datasets.xor_dataset()
#         targets = np.array([y for _, y in dataset])
#         learning_rate = 0.5
#         hidden_weights_loss = partial(
#             _loss_given, model, dataset, targets, "hidden_weights"
#         )
#
#         initial_loss = _current_loss(model, dataset, targets)
#
#         # Act
#         gradient = functional.numerical_gradient(
#             hidden_weights_loss, model.hidden_weights
#         )
#         model.hidden_weights = functional.gradient_descent_step(
#             model.hidden_weights, gradient, learning_rate
#         )
#         final_loss = _current_loss(model, dataset, targets)
#
#         # Assert
#         self.assertLess(final_loss, initial_loss)
#
#     def test_when_trained_for_a_few_steps_then_loss_decreases(self):
#         # Arrange
#         model = Xor(seed=42)
#         dataset = datasets.xor_dataset()
#         targets = np.array([y for _, y in dataset])
#         learning_rate = 0.5
#         epochs = 20
#
#         initial_loss = _current_loss(model, dataset, targets)
#
#         # Act
#         for _ in range(epochs):
#             for param_name in ("hidden_weights", "hidden_bias", "output_weights"):
#                 param = getattr(model, param_name)
#                 gradient = functional.numerical_gradient(
#                     partial(_loss_given, model, dataset, targets, param_name), param
#                 )
#                 setattr(
#                     model,
#                     param_name,
#                     functional.gradient_descent_step(param, gradient, learning_rate),
#                 )
#             bias_array = np.array([model.output_bias])
#             gradient = functional.numerical_gradient(
#                 partial(_loss_given_output_bias, model, dataset, targets),
#                 bias_array,
#             )
#             model.output_bias = functional.gradient_descent_step(
#                 bias_array, gradient, learning_rate
#             )[0]
#
#         final_loss = _current_loss(model, dataset, targets)
#
#         # Assert
#         self.assertLess(final_loss, initial_loss)
