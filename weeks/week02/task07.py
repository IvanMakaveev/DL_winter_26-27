import numpy as np

from dl_lib.metrics import functional


def main() -> int:
    y_true_balanced = np.array([1, 0, 1, 0, 1])
    y_pred_balanced = np.array([1, 0, 0, 1, 1])

    print("Balanced dataset:")
    print(f"  accuracy:  {functional.accuracy(y_true_balanced, y_pred_balanced):.4f}")
    print(f"  precision: {functional.precision(y_true_balanced, y_pred_balanced):.4f}")
    print(f"  recall:    {functional.recall(y_true_balanced, y_pred_balanced):.4f}")
    print(f"  f1:        {functional.f1_score(y_true_balanced, y_pred_balanced):.4f}")
    print()

    y_true_imbalanced = np.array([0] * 990 + [1] * 10)
    y_pred_imbalanced = np.array([0] * 990 + [1] * 7 + [0] * 3)

    print("Imbalanced dataset (990 negative, 10 positive):")
    print(
        f"  accuracy:  {functional.accuracy(y_true_imbalanced, y_pred_imbalanced):.4f}"
    )
    print(
        f"  precision: {functional.precision(y_true_imbalanced, y_pred_imbalanced):.4f}"
    )
    print(f"  recall:    {functional.recall(y_true_imbalanced, y_pred_imbalanced):.4f}")
    print(
        f"  f1:        {functional.f1_score(y_true_imbalanced, y_pred_imbalanced):.4f}"
    )

    return 0
