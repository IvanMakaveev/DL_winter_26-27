import torch

from dl_lib.optim.optimizers import SGD


def main() -> int:
    p = torch.tensor([1.0, 2.0])
    p.grad = torch.tensor([1.0, 2.0])
    optimizer = SGD([p], lr=0.01)

    print(f"Before step: p = {p}")
    optimizer.step()
    print(f"After step:  p = {p}")

    return 0
