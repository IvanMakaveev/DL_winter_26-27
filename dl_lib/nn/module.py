from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Module(ABC):
    training: bool = True

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma no cover
        pass

    def __call__(self, *args, **kwds) -> torch.Tensor:
        return self.forward(*args, **kwds)

    def train(self) -> Module:
        self.training = True
        return self

    def eval(self) -> Module:
        self.training = False
        return self

    def parameters(self) -> list[torch.Tensor]:
        return []
