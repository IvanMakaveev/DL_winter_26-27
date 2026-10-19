import matplotlib.pyplot as plt
import numpy as np

from dl_lib.losses import functional


def main() -> int:
    y_true = np.array([3.0, 5.0, 7.5, 9.0])
    y_pred_fixed = np.array([2.0, 6.0, 7.0, 9.0])

    growing_errors = np.linspace(0, 10, num=1000)
    growing_errors = np.insert(growing_errors, 500, 5)
    mae_values = []
    mse_values = []
    rmse_values = []

    for error in growing_errors:
        y_pred = y_pred_fixed.copy()
        y_pred[3] += error

        mae_values.append(functional.mae(y_true, y_pred))
        mse_values.append(functional.mse(y_true, y_pred))
        rmse_values.append(functional.rmse(y_true, y_pred))

        if error in [0, 5, 10]:
            print(
                f"Growing error={error:.1f}: MAE={mae_values[-1]:.4f}, MSE={mse_values[-1]:.4f}, RMSE={rmse_values[-1]:.4f}"
            )

    plt.figure(figsize=(10, 6))
    plt.plot(growing_errors, mae_values, label="MAE", color="blue")
    plt.plot(growing_errors, mse_values, label="MSE", color="orange")
    plt.plot(growing_errors, rmse_values, label="RMSE", color="green")
    plt.title("Loss Functions vs Growing Error")
    plt.xlabel("Growing Error")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return 0
