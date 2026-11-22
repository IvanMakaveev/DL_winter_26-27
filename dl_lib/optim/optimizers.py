import torch

from dl_lib.optim.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr, momentum=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p) for p in params]

    def step(self) -> None:
        with torch.no_grad():
            for p, v in zip(self.params, self.velocities):
                v *= self.momentum
                v += p.grad if p.grad is not None else 0.0
                p -= self.lr * v


class AdaGrad(Optimizer):
    def __init__(self, params, lr, eps=1e-10):
        super().__init__(params, lr)
        self.eps = eps
        self.accumulators = [torch.zeros_like(p) for p in params]

    def step(self) -> None:
        with torch.no_grad():
            for p, acc in zip(self.params, self.accumulators):
                acc += p.grad**2 if p.grad is not None else 0.0
                p -= (
                    self.lr * p.grad / (torch.sqrt(acc) + self.eps)
                    if p.grad is not None
                    else 0.0
                )
