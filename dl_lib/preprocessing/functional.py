import numpy as np


def min_max_normalize(arr: np.ndarray) -> np.ndarray:
    min_val = np.min(arr)
    max_val = np.max(arr)
    if np.isclose(max_val, min_val):
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def standardize(arr: np.ndarray) -> np.ndarray:
    mean = np.mean(arr)
    std = np.std(arr)
    if np.isclose(std, 0):
        return np.zeros_like(arr)
    return (arr - mean) / std
