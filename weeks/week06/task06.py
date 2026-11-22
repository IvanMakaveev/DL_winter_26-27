import matplotlib.pyplot as plt
import torch

from dl_lib.optim.optimizers import SGD, AdaGrad


def main() -> int:
    def f(x, y):
        return 10 * x**2 + 0.1 * y**2

    x1 = torch.tensor([4.0], requires_grad=True)
    y1 = torch.tensor([4.0], requires_grad=True)
    optimizer = SGD([x1, y1], lr=0.1, momentum=0.9)
    xs1 = []
    ys1 = []
    num_steps = 50
    for _ in range(num_steps):
        optimizer.zero_grad()
        loss = f(x1, y1)
        xs1.append(x1.item())
        ys1.append(y1.item())
        loss.backward()
        optimizer.step()

    print(f"SGD (momentum) final: x={x1.item():.4f}, y={y1.item():.4f}")
    x_vals = torch.linspace(-5, 5, 100)
    y_vals = torch.linspace(-5, 5, 100)
    X, Y = torch.meshgrid(x_vals, y_vals)
    Z = f(X, Y)
    plt.contourf(X, Y, Z, levels=30, cmap="Greys")
    plt.plot(xs1, ys1, color="orange", label="SGD (momentum=0.9)")
    plt.scatter(xs1, ys1, color="orange", s=7)

    x2 = torch.tensor([4.0], requires_grad=True)
    y2 = torch.tensor([4.0], requires_grad=True)
    optimizer = AdaGrad([x2, y2], lr=1)
    xs2 = []
    ys2 = []
    for _ in range(num_steps):
        optimizer.zero_grad()
        loss = f(x2, y2)
        xs2.append(x2.item())
        ys2.append(y2.item())
        loss.backward()
        optimizer.step()

    print(f"AdaGrad final: x={x2.item():.4f}, y={y2.item():.4f}")
    plt.plot(xs2, ys2, color="green", label="AdaGrad")
    plt.scatter(xs2, ys2, color="green", s=7)
    plt.title("SGD with momentum vs AdaGrad in a ravine")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()

    def f2(x):
        return 0.5 * x**2

    x3 = torch.tensor([5.0], requires_grad=True)
    optimizer = AdaGrad([x3], lr=0.3)
    xs3 = []
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = f2(x3)
        xs3.append(x3.item())
        loss.backward()
        if step == 0 and x3.grad is not None:
            x3.grad *= 50
        optimizer.step()
    print(f"AdaGrad (early spike) final x: {x3.item():.4f}")

    x4 = torch.tensor([5.0], requires_grad=True)
    optimizer = AdaGrad([x4], lr=0.3)
    xs4 = []
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = f2(x4)
        xs4.append(x4.item())
        loss.backward()
        optimizer.step()
    print(f"AdaGrad (no spike) final x: {x4.item():.4f}")

    steps = list(range(num_steps))
    plt.plot(steps, xs3, color="red", label="AdaGrad (early spike)")
    plt.plot(steps, xs4, color="green", label="AdaGrad (no spike)")
    plt.title("AdaGrad trajectories with and without an early gradient spike")
    plt.xlabel("step")
    plt.ylabel("x")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return 0
