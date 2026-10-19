import matplotlib.pyplot as plt
import numpy as np

from dl_lib.metrics import functional


def main() -> int:
    size = 100
    y_true = np.array([0] * (size // 2) + [1] * (size // 2))
    y_scores = np.concatenate([np.random.uniform(0, 1, size)])

    thresholds = np.linspace(0, 1, 50)
    precisions = []
    recalls = []

    for threshold in thresholds:
        y_pred = functional.apply_threshold(y_scores, threshold)
        precision = functional.precision(y_true, y_pred)
        recall = functional.recall(y_true, y_pred)
        precisions.append(precision)
        recalls.append(recall)

        if np.round(threshold, 2) in np.array([0.2, 0.5, 0.8]):
            print(
                f"threshold={threshold:.1f}: precision={precision:.4f}, recall={recall:.4f}"
            )

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label="Precision", color="blue")
    plt.plot(thresholds, recalls, label="Recall", color="orange")
    plt.xlabel("Threshold")
    plt.ylabel("Value")
    plt.title("Precision and Recall vs Threshold")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return 0
