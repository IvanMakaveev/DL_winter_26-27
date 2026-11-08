import torch

from dl_lib.nn.activations import Tanh


def main() -> int:
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    tanh = Tanh()

    print("x =", x)
    print("tanh(x) =", tanh(x))

    return 0
