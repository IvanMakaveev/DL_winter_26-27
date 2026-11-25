import torch

from dl_lib.nn.activations import Tanh
from dl_lib.nn.module import Module
from dl_lib.nn.sequential import Sequential


class FullyConnected(Module):
    def __init__(self, in_features, out_features, has_bias=True):
        super().__init__()
        self.has_bias = has_bias
        self.W = torch.randn(in_features, out_features, requires_grad=True)
        self.B = (
            torch.randn(out_features, requires_grad=True)
            if self.has_bias
            else torch.zeros(out_features)
        )

    def forward(self, x):
        return x @ self.W + self.B

    def parameters(self):
        return [self.W, self.B] if self.has_bias else [self.W]


def main() -> int:
    torch.manual_seed(42)

    mlp = Sequential(
        FullyConnected(10, 64),
        Tanh(),
        FullyConnected(64, 64),
        Tanh(),
        FullyConnected(64, 1),
    )

    print(
        f"Before eval: outer.training = {mlp.training}, inner.training = {mlp.modules[0].training}"
    )
    mlp.eval()
    print(
        f"After eval: outer.training = {mlp.training}, inner.training = {mlp.modules[0].training}"
    )
    mlp.train()
    print(
        f"After train: outer.training = {mlp.training}, inner.training = {mlp.modules[0].training}"
    )

    return 0
