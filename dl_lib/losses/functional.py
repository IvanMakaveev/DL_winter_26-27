import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    return np.mean(np.square(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    return np.sqrt(mse(y_true, y_pred))
