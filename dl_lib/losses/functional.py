import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    return np.mean(np.square(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    return np.sqrt(mse(y_true, y_pred))


def binary_cross_entropy(y_true: np.ndarray, probs: np.ndarray) -> np.float64:
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))


def categorical_cross_entropy(labels: np.ndarray, probs: np.ndarray) -> np.float64:
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return -np.mean(np.log(probs[np.arange(len(labels)), labels]))
