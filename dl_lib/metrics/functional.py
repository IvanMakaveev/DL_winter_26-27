import numpy as np


def apply_threshold(probs: np.ndarray, threshold: float) -> np.ndarray:
    return (probs >= threshold).astype(int)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    total = cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"]
    return (cm["tp"] + cm["tn"]) / total if total > 0 else 0.0


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    total_predicted_positive = cm["tp"] + cm["fp"]
    return cm["tp"] / total_predicted_positive if total_predicted_positive > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    total_actual_positive = cm["tp"] + cm["fn"]
    return cm["tp"] / total_actual_positive if total_actual_positive > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
