import unittest

import numpy as np

from dl_lib.losses import functional as losses_functional
from dl_lib.perceptron import functional


class TestInitializeWeights(unittest.TestCase):
    def test_when_called_then_returns_size_many_values_in_the_given_range(self):
        # Arrange
        low = 0.0
        high = 10.0
        size = 5

        # Act
        actual = functional.initialize_weights(low, high, size=size, seed=42)

        # Assert
        self.assertEqual(actual.shape, (size,))
        self.assertTrue(np.all(actual >= low))
        self.assertTrue(np.all(actual < high))

    def test_when_seed_is_fixed_then_result_is_reproducible(self):
        # Arrange
        low = 0.0
        high = 10.0
        seed = 42

        # Act
        first = functional.initialize_weights(low, high, size=3, seed=seed)
        second = functional.initialize_weights(low, high, size=3, seed=seed)

        # Assert
        np.testing.assert_array_equal(first, second)


class TestForward(unittest.TestCase):
    def test_when_activation_is_none_then_returns_the_weighted_sum_plus_bias(self):
        # Arrange
        x = np.array([2.0, 3.0])
        weights = np.array([1.0, 2.0])
        bias = 0.5
        expected = 8.5

        # Act
        actual = functional.forward(x, weights, bias=bias)

        # Assert
        self.assertAlmostEqual(actual, expected)

    def test_when_an_activation_is_given_then_applies_it_to_the_weighted_sum(self):
        # Arrange
        x = np.array([1.0])
        weights = np.array([0.0])
        bias = 0.0
        expected = 0.5

        # Act
        actual = functional.forward(
            x, weights, bias=bias, activation=lambda z: 1.0 / (1.0 + np.exp(-z))
        )

        # Assert
        self.assertAlmostEqual(actual, expected)


class TestDatasetLoss(unittest.TestCase):
    def test_when_called_then_matches_the_mean_squared_error_over_the_dataset(self):
        # Arrange
        dataset = [
            (np.array([0.0]), 0.0),
            (np.array([1.0]), 2.0),
            (np.array([2.0]), 4.0),
        ]
        targets = np.array([y for _, y in dataset])
        weights = np.array([1.0])
        bias = 0.0
        expected = (
            5.0 / 3.0
        )  # predictions [0, 1, 2] vs targets [0, 2, 4]: mean((0, 1, 4)) = 5/3

        # Act
        actual = functional.dataset_loss(
            dataset, targets, weights, bias, activation=None
        )

        # Assert
        self.assertAlmostEqual(actual, expected)

    def test_when_bce_loss_given_then_matches_the_binary_cross_entropy_over_the_dataset(
        self,
    ):
        # Arrange
        dataset = [
            (np.array([0.0, 0.0]), 0.0),
            (np.array([0.0, 1.0]), 0.0),
            (np.array([1.0, 0.0]), 0.0),
            (np.array([1.0, 1.0]), 1.0),
        ]
        targets = np.array([y for _, y in dataset])
        weights = np.array([0.5, -0.1])
        bias = 0.7
        activation = lambda z: 1.0 / (1.0 + np.exp(-z))
        predictions = np.array(
            [functional.forward(x, weights, bias, activation) for x, _ in dataset]
        )
        expected = losses_functional.binary_cross_entropy(targets, predictions)

        # Act
        actual = functional.dataset_loss(
            dataset,
            targets,
            weights,
            bias,
            activation=activation,
            loss=losses_functional.binary_cross_entropy,
        )

        # Assert
        self.assertAlmostEqual(actual, expected)


class TestDatasetLossGivenBias(unittest.TestCase):
    def test_when_called_then_matches_dataset_loss_with_the_unwrapped_bias(self):
        # Arrange
        dataset = [
            (np.array([0.0]), 0.0),
            (np.array([1.0]), 2.0),
            (np.array([2.0]), 4.0),
        ]
        targets = np.array([y for _, y in dataset])
        weights = np.array([2.0])
        bias_array = np.array([0.5])
        expected = functional.dataset_loss(
            dataset, targets, weights, bias_array[0], activation=None
        )

        # Act
        actual = functional.dataset_loss_given_bias(
            dataset, targets, weights, None, bias_array
        )

        # Assert
        self.assertAlmostEqual(actual, expected)


class TestNumericalGradient(unittest.TestCase):
    def test_when_called_on_a_quadratic_loss_then_matches_the_analytical_derivative(
        self,
    ):
        # Arrange
        def loss_fn(params: np.ndarray) -> float:
            return float(np.sum(params**2))

        params = np.array([3.0, -2.0])
        expected = np.array([6.0, -4.0])

        # Act
        actual = functional.numerical_gradient(loss_fn, params)

        # Assert
        np.testing.assert_allclose(actual, expected, atol=1e-4)

    def test_when_params_is_two_dimensional_then_gradient_matches_the_analytical_derivative(
        self,
    ):
        # Arrange
        def loss_fn(params: np.ndarray) -> float:
            return float(np.sum(params**2))

        params = np.array([[3.0, -2.0], [1.0, 0.5]])
        expected = np.array([[6.0, -4.0], [2.0, 1.0]])

        # Act
        actual = functional.numerical_gradient(loss_fn, params)

        # Assert
        np.testing.assert_allclose(actual, expected, atol=1e-4)

    def test_when_called_then_leaves_params_unchanged(self):
        # Arrange
        def loss_fn(params: np.ndarray) -> float:
            return float(np.sum(params**2))

        params = np.array([3.0, -2.0])
        expected = np.array([3.0, -2.0])

        # Act
        functional.numerical_gradient(loss_fn, params)

        # Assert
        np.testing.assert_array_equal(params, expected)


class TestGradientDescentStep(unittest.TestCase):
    def test_when_called_then_moves_params_against_the_gradient(self):
        # Arrange
        params = np.array([1.0, 1.0])
        gradient = np.array([2.0, -2.0])
        learning_rate = 0.1
        expected = np.array([0.8, 1.2])

        # Act
        actual = functional.gradient_descent_step(params, gradient, learning_rate)

        # Assert
        np.testing.assert_allclose(actual, expected)


class TestAnalyticalGradient(unittest.TestCase):
    def test_when_no_activation_then_matches_the_closed_form_gradient(self):
        # Arrange
        dataset = [
            (np.array([0.0]), 0.0),
            (np.array([1.0]), 2.0),
            (np.array([2.0]), 4.0),
        ]
        weights = np.array([1.0])
        expected_weight_gradient = np.array([-10.0 / 3.0])
        expected_bias_gradient = -2.0

        # Act
        actual_weight_gradient, actual_bias_gradient = functional.analytical_gradient(
            dataset, weights
        )

        # Assert
        np.testing.assert_allclose(actual_weight_gradient, expected_weight_gradient)
        self.assertAlmostEqual(actual_bias_gradient, expected_bias_gradient)

    def test_when_activation_given_then_matches_the_numerical_gradient(self):
        # Arrange
        dataset = [
            (np.array([0.0, 0.0]), 0.0),
            (np.array([0.0, 1.0]), 0.0),
            (np.array([1.0, 0.0]), 0.0),
            (np.array([1.0, 1.0]), 1.0),
        ]
        targets = np.array([y for _, y in dataset])
        weights = np.array([0.5, -0.1])
        bias = 0.7
        activation = lambda z: 1.0 / (1.0 + np.exp(-z))
        activation_derivative = lambda z: activation(z) * (1.0 - activation(z))

        weights_loss = lambda w: functional.dataset_loss(
            dataset, targets, w, bias, activation=activation
        )
        bias_loss = lambda b: functional.dataset_loss_given_bias(
            dataset, targets, weights, activation, np.array([b])
        )
        expected_weight_gradient = functional.numerical_gradient(weights_loss, weights)
        expected_bias_gradient = functional.numerical_gradient(
            bias_loss, np.array([bias])
        )[0]

        # Act
        actual_weight_gradient, actual_bias_gradient = functional.analytical_gradient(
            dataset,
            weights,
            bias,
            activation=activation,
            activation_derivative=activation_derivative,
        )

        # Assert
        np.testing.assert_allclose(
            actual_weight_gradient, expected_weight_gradient, atol=1e-4
        )
        self.assertAlmostEqual(actual_bias_gradient, expected_bias_gradient, places=4)

    def test_when_bce_loss_then_matches_the_numerical_gradient_without_activation_derivative(
        self,
    ):
        # Arrange
        dataset = [
            (np.array([0.0, 0.0]), 0.0),
            (np.array([0.0, 1.0]), 0.0),
            (np.array([1.0, 0.0]), 0.0),
            (np.array([1.0, 1.0]), 1.0),
        ]
        targets = np.array([y for _, y in dataset])
        weights = np.array([0.5, -0.1])
        bias = 0.7
        activation = lambda z: 1.0 / (1.0 + np.exp(-z))

        weights_loss = lambda w: functional.dataset_loss(
            dataset,
            targets,
            w,
            bias,
            activation=activation,
            loss=losses_functional.binary_cross_entropy,
        )
        bias_loss = lambda b: functional.dataset_loss_given_bias(
            dataset,
            targets,
            weights,
            activation,
            np.array([b]),
            loss=losses_functional.binary_cross_entropy,
        )
        expected_weight_gradient = functional.numerical_gradient(weights_loss, weights)
        expected_bias_gradient = functional.numerical_gradient(
            bias_loss, np.array([bias])
        )[0]

        # Act
        actual_weight_gradient, actual_bias_gradient = functional.analytical_gradient(
            dataset,
            weights,
            bias,
            activation=activation,
            loss="bce",
        )

        # Assert
        np.testing.assert_allclose(
            actual_weight_gradient, expected_weight_gradient, atol=1e-4
        )
        self.assertAlmostEqual(actual_bias_gradient, expected_bias_gradient, places=4)
