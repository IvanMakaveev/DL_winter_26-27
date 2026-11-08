import torch

from dl_lib.nn.module import Module


class DoubleModule(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0


def main() -> int:
    x = torch.tensor([1.0, 2.0, 3.0])
    double = DoubleModule()

    print("x =", x)
    print("double(x) =", double(x))

    return 0
