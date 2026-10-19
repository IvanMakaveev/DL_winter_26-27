import matplotlib.pyplot as plt
import numpy as np
import torch

from dl_lib.activations import functional


def main() -> int:
    x_values = np.array([-5, 0, 5])
    sigmoid_values = functional.sigmoid(x_values)
    relu_values = functional.relu(x_values)
    tanh_values = functional.tanh(x_values)

    x_plot = np.linspace(-10, 10, 100)
    sigmoid_range = functional.sigmoid(x_plot)
    relu_range = functional.relu(x_plot)
    tanh_range = functional.tanh(x_plot)

    print(f"sigmoid range: [{sigmoid_range.min():.4f}, {sigmoid_range.max():.4f}]")
    print(f"relu range:    [{relu_range.min():.4f}, {relu_range.max():.4f}]")
    print(f"tanh range:    [{tanh_range.min():.4f}, {tanh_range.max():.4f}]")

    for x, sig, rel, tan in zip(x_values, sigmoid_values, relu_values, tanh_values):
        print(f"At x={x}: sigmoid={sig:.4f}, relu={rel:.4f}, tanh={tan:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, sigmoid_range, label="sigmoid", color="blue")
    plt.plot(x_plot, relu_range, label="relu", color="green")
    plt.plot(x_plot, tanh_range, label="tanh", color="red")
    plt.title("Activation Functions")
    plt.xlabel("x")
    plt.ylabel("activation(x)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    torch_sigmoid = torch.sigmoid(torch.from_numpy(x_values)).numpy()
    torch_relu = torch.relu(torch.from_numpy(x_values)).numpy()
    torch_tanh = torch.tanh(torch.from_numpy(x_values)).numpy()

    print("Using PyTorch:")
    for x, sig, rel, tan in zip(x_values, torch_sigmoid, torch_relu, torch_tanh):
        print(f"At x={x}: sigmoid={sig:.4f}, relu={rel:.4f}, tanh={tan:.4f}")

    return 0
