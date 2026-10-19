import numpy as np
import torch

from dl_lib.losses import functional


def main() -> int:
    y_true = np.array([3.0, 5.0, 7.0, 9.0])
    y_pred = np.array([2.0, 6.0, 8.0, 7.0])

    mse_result = functional.mse(y_true, y_pred)
    torch_mse_result = torch.nn.functional.mse_loss(
        torch.from_numpy(y_pred), torch.from_numpy(y_true)
    ).item()

    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"MSE: {mse_result:.4f}")
    print(f"PyTorch MSE: {torch_mse_result:.4f}")
    print()

    y_pred_outlier = np.array([2.0, 6.0, 8.0, 100.0])

    mae_outlier = functional.mae(y_true, y_pred_outlier)
    mse_outlier = functional.mse(y_true, y_pred_outlier)

    print(f"y_pred with one extreme outlier: {y_pred_outlier}")
    print(f"MAE with outlier: {mae_outlier:.4f}")
    print(f"MSE with outlier: {mse_outlier:.4f}")

    return 0
