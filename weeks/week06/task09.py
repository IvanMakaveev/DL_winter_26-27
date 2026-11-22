import matplotlib.pyplot as plt
import torch

from dl_lib.optim.optimizers import AdamW


def main() -> int:
    def f(x):
        return 0.5 * x**2

    num_steps = 60

    x1 = torch.tensor([1.0], requires_grad=True)
    optimizer = AdamW([x1], lr=0.1, weight_decay=0.1)
    optimizer.v[0] = torch.ones_like(optimizer.v[0]) * 10
    xs1 = []
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = f(x1)
        xs1.append(x1.item())
        # loss.backward()
        optimizer.step()
    print(f"AdamW (large gradient history) final x: {x1.item():.4f}")

    x2 = torch.tensor([1.0], requires_grad=True)
    optimizer = AdamW([x2], lr=0.1, weight_decay=0.1)
    optimizer.v[0] = torch.ones_like(optimizer.v[0]) * 1 / 10
    xs2 = []
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = f(x2)
        xs2.append(x2.item())
        # loss.backward()
        optimizer.step()

    print(f"AdamW (small gradient history) final x: {x2.item():.4f}")

    steps = list(range(num_steps))
    plt.plot(steps, xs1, color="lightblue", label="AdamW (large gradient history)")
    plt.plot(steps, xs2, color="turquoise", linestyle="--", label="AdamW (small gradient history)")
    plt.title(
        "AdamW decays both parameters by the same proportion"
    )
    plt.xlabel("step")
    plt.ylabel("x")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return 0
