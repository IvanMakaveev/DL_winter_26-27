import numpy as np


def main() -> int:
    y_true = np.array([3, 5, 7, 9])
    y_pred = np.array([2, 6, 8, 7])

    print(f"y_true:          {y_true}")
    print(f"y_pred:          {y_pred}")

    errors = y_pred - y_true
    absolute_errors = np.abs(errors)
    squared_errors = errors**2

    print(f"errors:          {errors}")
    print(f"absolute errors: {absolute_errors}")
    print(f"squared errors:  {squared_errors}")
    print()

    mean_err = np.mean(errors)
    mae = np.mean(absolute_errors)
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)

    print(f"mean error: {mean_err:.4f}")
    print(f"MAE (mean absolute error): {mae:.4f}")
    print(f"MSE (mean squared error):  {mse:.4f}")
    print(f"RMSE (root mean squared error): {rmse:.4f}")

    return 0
