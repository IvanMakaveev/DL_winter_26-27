import matplotlib.pyplot as plt
import torch

from dl_lib.nn.activations import Tanh
from dl_lib.nn.module import Module
from dl_lib.nn.sequential import Sequential
from dl_lib.optim.optimizer import Optimizer
from dl_lib.optim.optimizers import SGD


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


def mse_loss(predicted: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.square(predicted - targets))


def train(
    model: Module,
    optimizer: Optimizer,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    test_X: torch.Tensor,
    test_Y: torch.Tensor,
    epochs: int,
) -> tuple:

    train_losses = []
    test_losses = []
    train_losses.append(evaluate(model, train_X, train_Y))
    test_losses.append(evaluate(model, test_X, test_Y))

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        y_hat = model(train_X)
        loss = mse_loss(y_hat, train_Y)
        loss.backward()
        optimizer.step()

        train_losses.append(evaluate(model, train_X, train_Y))
        test_losses.append(evaluate(model, test_X, test_Y))

    return train_losses, test_losses


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
    Noise = torch.randn(n, 1, generator=data_generator) * 0.1

    Y = X @ true_w + Noise

    return X, Y


def main() -> int:
    true_w_gen = torch.Generator().manual_seed(1337)
    true_w = torch.randn(10, 1, generator=true_w_gen)
    data_generator = torch.Generator().manual_seed(42)

    torch.manual_seed(42)

    mlp = Sequential(
        FullyConnected(10, 64),
        Tanh(),
        FullyConnected(64, 64),
        Tanh(),
        FullyConnected(64, 1),
    )

    x_train, y_train = generate_dataset(20, true_w, data_generator)
    x_test, y_test = generate_dataset(200, true_w, data_generator)

    optimizer = SGD(mlp.parameters(), lr=0.01)

    train_losses, test_losses = train(
        mlp, optimizer, x_train, y_train, x_test, y_test, 300
    )
    print(
        f"Final losses: train loss = {train_losses[-1]}, test loss = {test_losses[-1]}"
    )

    plt.plot(range(len(train_losses)), train_losses, label="Train Loss (20 examples)")
    plt.plot(range(len(test_losses)), test_losses, label="Test Loss (200 examples)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.title("A plain MLP overfits a tiny train set")
    plt.tight_layout()
    plt.show()

    return 0
