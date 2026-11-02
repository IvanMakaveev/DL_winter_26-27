from collections.abc import Callable
from typing import Any

import numpy as np

from dl_lib.losses.functional import mse


def initialize_weights(
    low: float, high: float, size: int, seed: int | None = None
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.uniform(low, high, size)


def forward(
    inputs: np.ndarray,
    weights: np.ndarray,
    bias: float = 0.0,
    activation: Callable[..., Any] | None = None,
) -> float:
    result = np.dot(inputs, weights) + bias
    if activation is not None:
        result = activation(result)
    return result


def numerical_gradient(
    loss: Callable[..., float], parameters: np.ndarray, epsilon: float = 1e-4
) -> np.ndarray:
    gradients = np.zeros_like(parameters)
    # It should work when parameters is 1D or 2D array
    for i in range(parameters.size):
        original_value = parameters.flat[i]
        parameters.flat[i] = original_value + epsilon
        loss_plus_epsilon = loss(parameters)
        parameters.flat[i] = original_value - epsilon
        loss_minus_epsilon = loss(parameters)
        parameters.flat[i] = original_value
        gradients.flat[i] = (loss_plus_epsilon - loss_minus_epsilon) / (2 * epsilon)

    return gradients


def gradient_descent_step(
    parameters: np.ndarray, gradients: np.ndarray, learning_rate: float
) -> np.ndarray:
    return parameters - learning_rate * gradients


def dataset_loss(
    dataset: list[tuple[float, float]],
    targets: np.ndarray,
    weights: np.ndarray,
    bias: float = 0.0,
    activation: Callable[..., Any] | None = None,
    loss: Callable[..., float] | None = mse,
) -> float:
    inputs = [np.atleast_1d(x) for x, _ in dataset]
    predicted = np.array(
        [forward(x, weights, bias, activation) for x in inputs]
    ).reshape(-1)
    return loss(targets, predicted)


def analytical_gradient(
    dataset: list[tuple[float, float]],
    weights: np.ndarray,
    bias: float = 0.0,
    activation: Callable[..., Any] | None = None,
    activation_derivative: Callable[..., Any] | None = None,
) -> np.ndarray:
    inputs = [np.atleast_1d(x) for x, _ in dataset]
    targets = np.array([y for _, y in dataset])
    predicted = np.array(
        [forward(x, weights, bias, activation) for x in inputs]
    ).reshape(-1)
    errors = predicted - targets

    if activation_derivative:
        errors *= activation_derivative(predicted)

    weight_grad = 2 * np.dot(errors, inputs) / len(dataset)
    bias_grad = 2 * np.sum(errors) / len(dataset)

    return weight_grad, bias_grad
