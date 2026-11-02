from dl_lib.losses.functional import mse
import numpy as np


def initialize_weights(low: float, high: float, size: int, seed: int = 42) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.uniform(low, high, size)

def forward(inputs: np.ndarray, weights: np.ndarray, bias:float = 0.0, activation: callable = None) -> float:
    result = np.dot(inputs, weights) + bias
    if activation:
        result = activation(result)
    return result

def numerical_gradient(loss: callable, parameters: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    gradients = np.zeros_like(parameters)
    for i in range(len(parameters)):
        original_value = parameters[i]
        parameters[i] = original_value + epsilon
        loss_plus = loss(parameters)
        parameters[i] = original_value - epsilon
        loss_minus = loss(parameters)
        gradients[i] = (loss_plus - loss_minus) / (2 * epsilon)
        parameters[i] = original_value
    return gradients

def gradient_descent_step(parameters: np.ndarray, gradients: np.ndarray, learning_rate: float) -> np.ndarray:
    return parameters - learning_rate * gradients

def dataset_loss(dataset: list[tuple[float, float]], targets: np.ndarray, weights: np.ndarray, bias: float = 0.0, activation: callable = None, loss: callable = mse) -> float:
    inputs = np.array([x for x, _ in dataset])
    predicted = np.array([forward(np.array([x]), weights, bias, activation) for x in inputs])
    return loss(predicted, targets)

def analytical_gradient(dataset: list[tuple[float, float]], weights: np.ndarray, bias: float = 0.0, activation: callable = None, activation_derivative: callable = None) -> np.ndarray:
    inputs = np.array([x for x, _ in dataset])
    targets = np.array([y for _, y in dataset])
    predicted = np.array([forward(np.array([x]), weights, bias, activation) for x in inputs])
    errors = predicted - targets
    
    if activation_derivative:
        errors *= activation_derivative(predicted)

    weight_grad = 2 * np.dot(errors, inputs) / len(dataset)
    bias_grad = 2 * np.sum(errors) / len(dataset)

    if weight_grad.ndim == 0:
        weight_grad = np.array([weight_grad])

    return weight_grad, bias_grad 