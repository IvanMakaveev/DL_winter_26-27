import numpy as np

from dl_lib.metrics import functional


def main() -> int:
    y_true = np.array([0] * 990 + [1] * 10)
    y_pred = np.zeros(1000)

    print("Dataset: 990 negative, 10 positive")
    print("Model: always predicts negative, never looks at the input")
    print()

    print(f"accuracy:  {functional.accuracy(y_true, y_pred):.4f}")
    print(f"precision: {functional.precision(y_true, y_pred):.4f}")
    print(f"recall:    {functional.recall(y_true, y_pred):.4f}")
    print(f"f1:        {functional.f1_score(y_true, y_pred):.4f}")

    return 0
