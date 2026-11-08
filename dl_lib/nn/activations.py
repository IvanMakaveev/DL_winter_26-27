import torch

from dl_lib.nn.module import Module


class Sigmoid(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + torch.exp(-x))


class Tanh(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


class ReLU(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.maximum(x, torch.zeros_like(x))


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, self.negative_slope * x)
