from abc import ABC, abstractmethod

import torch


class Optimizer(ABC):
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    @abstractmethod
    def step(self):  # pragma: no cover
        pass

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = torch.zeros_like(p.grad)
