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


class RMSprop(Optimizer):
    def __init__(self, params, lr, alpha=0.99, eps=1e-8):
        super().__init__(params, lr)
        self.alpha = alpha
        self.eps = eps
        self.accumulators = [torch.zeros_like(p) for p in params]

    def step(self) -> None:
        with torch.no_grad():
            for p, acc in zip(self.params, self.accumulators):
                acc *= self.alpha
                acc += (1 - self.alpha) * (p.grad**2 if p.grad is not None else 0.0)
                p -= (
                    self.lr * p.grad / (torch.sqrt(acc) + self.eps)
                    if p.grad is not None
                    else 0.0
                )


class Adam(Optimizer):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        with torch.no_grad():
            for p, m, v in zip(self.params, self.m, self.v):
                decayed_grad = (
                    p.grad if p.grad is not None else 0.0
                ) + self.weight_decay * p
                m *= self.beta1
                m += (1 - self.beta1) * decayed_grad
                v *= self.beta2
                v += (1 - self.beta2) * (decayed_grad**2)
                m_hat = m / (1 - self.beta1**self.t)
                v_hat = v / (1 - self.beta2**self.t)
                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, beta1, beta2, eps, weight_decay)

    def step(self) -> None:
        self.t += 1
        with torch.no_grad():
            for p, m, v in zip(self.params, self.m, self.v):
                grad = p.grad if p.grad is not None else 0.0
                m *= self.beta1
                m += (1 - self.beta1) * grad
                v *= self.beta2
                v += (1 - self.beta2) * (grad**2)
                m_hat = m / (1 - self.beta1**self.t)
                v_hat = v / (1 - self.beta2**self.t)
                p -= self.lr * self.weight_decay * p
                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
