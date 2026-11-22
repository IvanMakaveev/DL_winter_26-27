import matplotlib.pyplot as plt
import torch

from dl_lib.optim.optimizers import AdaGrad, RMSprop


def main() -> int:
    def f(x):
        return 0.5 * x**2

    num_steps = 150

    x1 = torch.tensor([5.0], requires_grad=True)
    optimizer = AdaGrad([x1], lr=0.5)
    xs1 = []
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = f(x1)
        xs1.append(x1.item())
        loss.backward()
        if step == 0 and x1.grad is not None:
            x1.grad *= 50
        optimizer.step()
    print(f"AdaGrad (early spike) final x: {x1.item():.4f}")

    x2 = torch.tensor([5.0], requires_grad=True)
    optimizer = RMSprop([x2], lr=0.5)
    xs2 = []
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = f(x2)
        xs2.append(x2.item())
        loss.backward()
        if step == 0 and x2.grad is not None:
            x2.grad *= 50
        optimizer.step()
    print(f"RMSprop (early spike) final x: {x2.item():.4f}")

    steps = list(range(num_steps))
    plt.plot(steps, xs1, color="green", label="AdaGrad (early spike)")
    plt.plot(steps, xs2, color="purple", label="RMSprop (early spike)")
    plt.title("RMSprop recovers from an early gradient spike")
    plt.xlabel("step")
    plt.ylabel("x")
    plt.legend()
    plt.tight_layout()
    plt.show()

    num_steps = 100

    x3 = torch.tensor([5.0], requires_grad=True)
    optimizer = RMSprop([x3], lr=0.1)
    xs3 = []
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = f(x3)
        xs3.append(x3.item())
        loss.backward()
        optimizer.step()
    print(f"RMSprop (clean gradient) final x: {x3.item():.4f}")

    x4 = torch.tensor([5.0], requires_grad=True)
    optimizer = RMSprop([x4], lr=0.1)
    xs4 = []
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = f(x4)
        xs4.append(x4.item())
        loss.backward()
        if x4.grad is not None:
            x4.grad += torch.randn_like(x4.grad)
        optimizer.step()
    print(f"RMSprop (noisy gradient) final x: {x4.item():.4f}")

    steps = list(range(num_steps))
    plt.plot(steps, xs3, color="purple", label="RMSprop (clean gradient)")
    plt.plot(steps, xs4, color="red", label="RMSprop (noisy gradient)")
    plt.title("RMSprop with clean vs noisy gradients")
    plt.xlabel("step")
    plt.ylabel("x")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return 0
