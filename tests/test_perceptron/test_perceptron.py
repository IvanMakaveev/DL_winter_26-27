import unittest
from functools import partial

import numpy as np

from dl_lib.activations import functional as activations_functional
from dl_lib.perceptron import datasets, functional
from dl_lib.perceptron.perceptron import Perceptron


class TestInit(unittest.TestCase):
    def test_when_use_bias_is_false_then_bias_is_zero(self):
        # Arrange
        n_inputs = 2
        expected_bias = 0.0

        # Act
        actual = Perceptron(n_inputs, use_bias=False, seed=42)

        # Assert
        self.assertEqual(actual.weights.shape, (n_inputs,))
        self.assertFalse(actual.use_bias)
        self.assertEqual(actual.bias, expected_bias)

    def test_when_use_bias_is_true_then_bias_is_drawn_from_the_given_range(self):
        # Arrange
        n_inputs = 2
        low = 0.0
        high = 10.0

        # Act
        actual = Perceptron(n_inputs, low=low, high=high, use_bias=True, seed=42)

        # Assert
        self.assertTrue(actual.use_bias)
        self.assertGreaterEqual(actual.bias, low)
        self.assertLess(actual.bias, high)

    def test_when_seed_is_fixed_then_weights_are_reproducible(self):
        # Arrange
        seed = 42

        # Act
        first = Perceptron(3, seed=seed)
        second = Perceptron(3, seed=seed)

        # Assert
        np.testing.assert_array_equal(first.weights, second.weights)


class TestForward(unittest.TestCase):
    def test_when_called_then_output_is_finite(self):
        # Arrange
        model = Perceptron(2, use_bias=True, seed=42)
        x = np.array([1.0, 0.0])

        # Act
        actual = model.forward(x)

        # Assert
        self.assertTrue(np.isfinite(actual))

    def test_when_no_activation_then_matches_the_plain_weighted_sum(self):
        # Arrange
        model = Perceptron(2, use_bias=True, seed=42)
        x = np.array([1.0, 2.0])
        expected = functional.forward(x, model.weights, model.bias)

        # Act
        actual = model.forward(x)

        # Assert
        self.assertAlmostEqual(actual, expected)

    def test_when_an_activation_is_given_then_applies_it(self):
        # Arrange
        model = Perceptron(
            2, use_bias=True, activation=activations_functional.sigmoid, seed=42
        )
        x = np.array([1.0, 2.0])

        # Act
        actual = model.forward(x)

        # Assert
        self.assertGreater(actual, 0.0)
        self.assertLess(actual, 1.0)


class TestTraining(unittest.TestCase):
    def test_when_a_single_gradient_descent_step_is_applied_then_loss_decreases(self):
        # Arrange
        model = Perceptron(2, low=-1.0, high=1.0, use_bias=True, seed=42)
        dataset = datasets.and_dataset()
        targets = np.array([y for _, y in dataset])
        learning_rate = 0.1
        weights_loss = partial(
            functional.dataset_loss,
            dataset,
            targets,
            bias=model.bias,
            activation=model.activation,
        )

        initial_loss = weights_loss(model.weights)

        # Act
        weight_gradient = functional.numerical_gradient(weights_loss, model.weights)
        model.weights = functional.gradient_descent_step(
            model.weights, weight_gradient, learning_rate
        )
        final_loss = weights_loss(model.weights)

        # Assert
        self.assertLess(final_loss, initial_loss)

    def test_when_trained_for_a_few_steps_then_loss_decreases(self):
        # Arrange
        model = Perceptron(2, low=-1.0, high=1.0, use_bias=True, seed=42)
        dataset = datasets.and_dataset()
        targets = np.array([y for _, y in dataset])
        learning_rate = 0.1
        epochs = 50

        initial_loss = functional.dataset_loss(
            dataset, targets, model.weights, model.bias, model.activation
        )

        # Act
        for _ in range(epochs):
            weights_loss = partial(
                functional.dataset_loss,
                dataset,
                targets,
                bias=model.bias,
                activation=model.activation,
            )
            weight_gradient = functional.numerical_gradient(weights_loss, model.weights)
            model.weights = functional.gradient_descent_step(
                model.weights, weight_gradient, learning_rate
            )

            bias_loss = partial(
                functional.dataset_loss_given_bias,
                dataset,
                targets,
                model.weights,
                model.activation,
            )
            bias_array = np.array([model.bias])
            bias_gradient = functional.numerical_gradient(bias_loss, bias_array)
            model.bias = functional.gradient_descent_step(
                bias_array, bias_gradient, learning_rate
            )[0]

        final_loss = functional.dataset_loss(
            dataset, targets, model.weights, model.bias, model.activation
        )

        # Assert
        self.assertLess(final_loss, initial_loss)
