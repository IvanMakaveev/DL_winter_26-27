import random

import torch

from dl_lib.optim.optimizers import SGD


def main() -> int:
    def f(x):
        return x**4 + x**3 - 5 * x**2

    def run(lr, momentum):
        x = torch.tensor([2.0], requires_grad=True)
        optimizer = SGD([x], lr=lr, momentum=momentum)
        for _ in range(1000):
            optimizer.zero_grad()
            loss = f(x)
            loss.backward()
            optimizer.step()

        return x.item(), f(x).item()

    print(f"{'lr':>10} {'momentum':>10} {'final x':>10} {'f(x)':>10}")
    best_config = None
    random.seed(42)
    for _ in range(8):
        lr = random.uniform(0.001, 0.05)
        momentum = random.uniform(0.0, 1.2)
        x, fx = run(lr, momentum)
        print(f"{lr:>10.4f} {momentum:>10.4f} {x:>10.4f} {fx:>10.4f}")
        if best_config is None or fx < best_config[3]:
            best_config = (lr, momentum, x, fx)

    if best_config is not None:
        lr, momentum, x, fx = best_config
        print(
            f"Best config: lr={lr:.4f}, momentum={momentum:.4f} -> x={x:.4f}, f(x)={fx:.4f}"
        )
    return 0
