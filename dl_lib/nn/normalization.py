import torch

from dl_lib.nn.module import Module


class BatchNorm1d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.gamma = torch.ones(num_features, requires_grad=True)
        self.beta = torch.zeros(num_features, requires_grad=True)
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * mean
                var_unb = x.var(dim=0, unbiased=True)
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * var_unb
        else:
            mean = self.running_mean
            var = self.running_var

        return (self.gamma * (x - mean)) / (torch.sqrt(var + self.eps)) + self.beta

    def parameters(self):
        return [self.gamma, self.beta]


class LayerNorm(Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = torch.ones(normalized_shape, requires_grad=True)
        self.beta = torch.zeros(normalized_shape, requires_grad=True)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return (self.gamma * (x - mean)) / (torch.sqrt(var + self.eps)) + self.beta

    def parameters(self):
        return [self.gamma, self.beta]
