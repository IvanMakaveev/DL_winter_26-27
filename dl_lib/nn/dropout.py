import torch

from dl_lib.nn.module import Module


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0 <= p <= 1):
            raise ValueError("Dropout probability must be in the range [0, 1]")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x

        if self.p == 1:
            return x * 0

        mask = torch.rand_like(x) > self.p
        return mask * x / (1 - self.p)
