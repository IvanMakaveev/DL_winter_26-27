import torch

from dl_lib.nn.activations import Tanh
from dl_lib.nn.dropout import Dropout
from dl_lib.nn.module import Module
from dl_lib.nn.normalization import BatchNorm1d, LayerNorm
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


def r2_score(predicted: torch.Tensor, targets: torch.Tensor) -> float:
    rss = torch.sum(torch.square(predicted - targets))
    tss = torch.sum(torch.square(targets - torch.mean(targets)))
    return (1 - rss / tss).item()


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

    for _ in range(epochs):
        model.train()
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

    x_train, y_train = generate_dataset(20, true_w, data_generator)
    x_test, y_test = generate_dataset(200, true_w, data_generator)

    models = [
        (
            "plain MLP",
            Sequential(
                FullyConnected(10, 64),
                Tanh(),
                FullyConnected(64, 64),
                Tanh(),
                FullyConnected(64, 1),
            ),
            "no regularization, overfits the 20-example train set",
        ),
        (
            "+Dropout",
            Sequential(
                FullyConnected(10, 64),
                Tanh(),
                Dropout(),
                FullyConnected(64, 64),
                Tanh(),
                Dropout(),
                FullyConnected(64, 1),
            ),
            "randomly zeroes activations, narrows the train/test gap",
        ),
        (
            "+BatchNorm1d",
            Sequential(
                FullyConnected(10, 64),
                BatchNorm1d(64),
                Tanh(),
                FullyConnected(64, 64),
                BatchNorm1d(64),
                Tanh(),
                FullyConnected(64, 1),
            ),
            "normalizes over the batch, mainly a training stabilizer here",
        ),
        (
            "+LayerNorm",
            Sequential(
                FullyConnected(10, 64),
                LayerNorm(64),
                Tanh(),
                FullyConnected(64, 64),
                LayerNorm(64),
                Tanh(),
                FullyConnected(64, 1),
            ),
            "normalizes per sample, stable even with this tiny batch",
        ),
    ]

    results = [
        (
            "baseline",
            r2_score(torch.full_like(y_test, y_train.mean().item()), y_test),
            "mean predictor, ignores x",
        )
    ]
    for name, model, comment in models:
        optimizer = SGD(model.parameters(), lr=0.01)
        train(model, optimizer, x_train, y_train, x_test, y_test, 300)
        model.eval()
        with torch.no_grad():
            results.append((name, r2_score(model(x_test), y_test), comment))

    print(f"{'model':<17}{'test r2':>8}  comment")
    for name, r2score, comment in results:
        print(f"{name:<17}{r2score:>8.4f}  {comment}")

    best_name, best_r2score, _ = max(results, key=lambda row: row[1])
    print(f"Best model: {best_name} (test r2 = {best_r2score:.4f})")

    return 0
