import matplotlib.pyplot as plt
import torch

from dl_lib.nn.activations import Tanh
from dl_lib.nn.module import Module
from dl_lib.nn.normalization import BatchNorm1d
from dl_lib.nn.sequential import Sequential
from dl_lib.optim.optimizer import Optimizer
from dl_lib.optim.optimizers import SGD


class FullyConnected(Module):
    def __init__(self, in_features, out_features, has_bias=True):
        super().__init__()
        self.has_bias = has_bias
        self.W = (torch.randn(in_features, out_features) * 0.1).requires_grad_(True)
        self.B = (
            torch.randn(out_features, requires_grad=True)
            if self.has_bias
            else torch.zeros(out_features)
        )

    def forward(self, x):
        return x @ self.W + self.B

    def parameters(self):
        return [self.W, self.B] if self.has_bias else [self.W]


def mse_loss(predicted: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.square(predicted - targets))


def train(
    model: Sequential,
    optimizer: Optimizer,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    epochs: int,
) -> tuple:

    train_losses = []
    train_losses.append(evaluate(model, train_X, train_Y))
    gradients = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_hat = model(train_X)
        loss = mse_loss(y_hat, train_Y)
        loss.backward()
        gradients.append(
            torch.stack(
                [
                    (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
                    for p in module.parameters()
                ]
            )
            for module in model.modules
            if isinstance(module, FullyConnected)
        )
        optimizer.step()

        train_losses.append(evaluate(model, train_X, train_Y))

    return train_losses, gradients


def evaluate(model: Module, test_X: torch.Tensor, test_Y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        Y_predicted = model(test_X)
        loss = mse_loss(Y_predicted, test_Y)

    return loss.item()


def generate_dataset(
    n: int, true_w: torch.Tensor, data_generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.randn(n, 10, generator=data_generator)
    Y = X @ true_w
    return X, Y


def main() -> int:
    batchnorm = BatchNorm1d(3)
    x = torch.tensor(
        [
            [4.2, -1.2, 2.7],
            [1.8, 2.4, -3.5],
            [-1.0, 0.5, 0.8],
            [3.4, -2.1, 1.4],
        ]
    )
    w = torch.rand(3)
    y = w * x
    batchnorm.eval()
    print(f"batchnorm.gamma = {batchnorm.gamma}")
    print(f"batchnorm.beta = {batchnorm.beta}")
    print(f"x.mean(dim=0) ={x.mean(dim=0)}")
    print(f"x.var(dim=0, unbiased=False) = {x.var(dim=0, unbiased=False)}")
    print(f"batchnorm(x).mean(dim=0) = {batchnorm(x).mean(dim=0)}")
    print(
        f"batchnorm(x).var(dim=0, unbiased=False) = {batchnorm(x).var(dim=0, unbiased=False)}"
    )
    print(
        f"Before any forward call: batchnorm.running_mean = {batchnorm.running_mean}, batchnorm.running_var = {batchnorm.running_var}"
    )
    optimizer = SGD(batchnorm.parameters(), 0.01)
    batchnorm.train()
    optimizer.zero_grad()
    y_hat = batchnorm(x)
    loss = mse_loss(y_hat, y)
    loss.backward()
    optimizer.step()
    batchnorm.eval()

    print(
        f"After one training call: batchnorm.running_mean = {batchnorm.running_mean}, batchnorm.running_var = {batchnorm.running_var}"
    )
    print(f"Evaluating: {batchnorm(x).mean(dim=0) = }")

    true_w_gen = torch.Generator().manual_seed(1337)
    true_w = torch.randn(10, 1, generator=true_w_gen)
    data_generator = torch.Generator().manual_seed(42)

    torch.manual_seed(42)

    mlp = Sequential(FullyConnected(10, 32, False), BatchNorm1d(32), Tanh())
    for _ in range(11):
        mlp.append(FullyConnected(32, 32, False))
        mlp.append(BatchNorm1d(32))
        mlp.append(Tanh())

    mlp.append(FullyConnected(32, 1, False))

    x_train, y_train = generate_dataset(128, true_w, data_generator)

    optimizer = SGD(mlp.parameters(), lr=0.01)

    train_losses, gradients = train(mlp, optimizer, x_train, y_train, 50)

    print(
        f"With BatchNorm1d: loss[0] = {train_losses[0]:.4f}, loss[25] = {train_losses[25]:.4f}, loss[-1] = {train_losses[-1]:.4f}"
    )
    grad_averages = [torch.mean(torch.abs(grad)).item() for grad in gradients[-1]]
    print(f"With BatchNorm1d: gradient magnitude per layer = {grad_averages}")

    plt.plot(range(1, len(grad_averages) + 1), grad_averages, marker="o")
    plt.title("BatchNorm1d recovers gradient flow across all 12 layers")
    plt.xlabel("Layer (1 = first, 12 = last)")
    plt.ylabel("Mean Gradient (log scale)")
    plt.yscale("log")
    plt.xticks(range(1, len(grad_averages) + 1))
    plt.tight_layout()
    plt.show()

    return 0
