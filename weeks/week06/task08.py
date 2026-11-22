import matplotlib.pyplot as plt
import torch

from dl_lib.optim.optimizers import RMSprop, Adam


def main() -> int:
    def f(x):
        return 0.5 * x**2

    num_steps = 100

    x1 = torch.tensor([5.0], requires_grad=True)
    optimizer = RMSprop([x1], lr=0.1)
    xs1 = []
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = f(x1)
        xs1.append(x1.item())
        loss.backward()
        if x1.grad is not None:
            x1.grad += torch.randn_like(x1.grad)
        optimizer.step()
    print(f"RMSprop (noisy gradient) final x: {x1.item():.4f}")

    x2 = torch.tensor([5.0], requires_grad=True)
    optimizer = Adam([x2], lr=0.1)
    xs2 = []
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = f(x2)
        xs2.append(x2.item())
        loss.backward()
        if x2.grad is not None:
            x2.grad += torch.randn_like(x2.grad)
        optimizer.step()

    print(f"Adam (noisy gradient) final x: {x2.item():.4f}")

    steps = list(range(num_steps))
    plt.plot(steps, xs1, color="red", label="RMSprop (noisy gradient)")
    plt.plot(steps, xs2, color="lightblue", label="Adam (noisy gradient)")
    plt.title("Adam's momentum smooths out the noise in the gradient")
    plt.xlabel("step")
    plt.ylabel("x")
    plt.legend()
    plt.tight_layout()
    plt.show()

    num_steps = 60

    x3 = torch.tensor([1.0], requires_grad=True)
    optimizer = Adam([x3], lr=0.1, weight_decay=0.1)
    optimizer.v[0] = torch.ones_like(optimizer.v[0]) * 10
    xs3 = []
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = f(x3)
        xs3.append(x3.item())
        # loss.backward()
        optimizer.step()
    print(f"Adam (large gradient history) final x: {x3.item():.4f}")

    x4 = torch.tensor([1.0], requires_grad=True)
    optimizer = Adam([x4], lr=0.1, weight_decay=0.1)
    optimizer.v[0] = torch.ones_like(optimizer.v[0]) * 1 / 10
    xs4 = []
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = f(x4)
        xs4.append(x4.item())
        # loss.backward()
        optimizer.step()

    print(f"Adam (small gradient history) final x: {x4.item():.4f}")

    steps = list(range(num_steps))
    plt.plot(steps, xs3, color="lightblue", label="Adam (large gradient history)")
    plt.plot(steps, xs4, color="turquoise", label="Adam (small gradient history)")
    plt.title("Adam decays the same gradient history differently based on its magnitude")
    plt.xlabel("step")
    plt.ylabel("x")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return 0
