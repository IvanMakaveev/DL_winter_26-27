import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)
