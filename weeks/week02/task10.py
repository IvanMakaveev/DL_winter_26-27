import matplotlib.pyplot as plt
import numpy as np
import torch

from dl_lib.activations import functional


def main() -> int:
    x_values = np.array([-10, 0, 10])
    sigmoid_values = functional.sigmoid(x_values)

    for x, sig in zip(x_values, sigmoid_values):
        print(f"sigmoid({x}) = {sig:.6f}")

    output_range = (sigmoid_values.min(), sigmoid_values.max())
    print(f"Output range: [{output_range[0]:.6f}, {output_range[1]:.6f}]")

    x_plot = np.linspace(-10, 10, 100)
    y_plot = functional.sigmoid(x_plot)

    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, y_plot, label="Sigmoid Function", color="blue")
    plt.title("Sigmoid")
    plt.xlabel("x")
    plt.ylabel("sigmoid(x)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    torch_sigmoid_values = torch.sigmoid(torch.from_numpy(x_values)).numpy()
    print("Using PyTorch:")
    for x, sig in zip(x_values, torch_sigmoid_values):
        print(f"torch.sigmoid({x}) = {sig:.6f}")

    return 0
