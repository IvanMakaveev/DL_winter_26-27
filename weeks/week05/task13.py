import torch

from dl_lib.nn.activations import LeakyReLU


def main() -> int:
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    leaky_relu = LeakyReLU(negative_slope=0.1)

    print("x =", x)
    print("leaky_relu(x) =", leaky_relu(x))

    return 0
