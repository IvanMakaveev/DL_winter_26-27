import numpy as np


def apply_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (probs >= threshold).astype(int)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
