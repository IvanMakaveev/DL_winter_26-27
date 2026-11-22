import matplotlib.pyplot as plt
import torch

from dl_lib.optim.optimizers import SGD


def main() -> int:
    def f(x):
        return x**4 + x**3 - 5 * x**2

    x1 = torch.tensor([2.0], requires_grad=True)
    x2 = torch.tensor([2.0], requires_grad=True)
    optimizer_no_momentum = SGD([x1], lr=0.02, momentum=0.0)
    optimizer_with_momentum = SGD([x2], lr=0.02, momentum=0.9)

    num_steps = 1000
    xs1 = []
    ys1 = []
    xs2 = []
    ys2 = []
    for _ in range(num_steps):
        optimizer_no_momentum.zero_grad()
        loss1 = f(x1)
        xs1.append(x1.item())
        ys1.append(loss1.item())
        loss1.backward()
        optimizer_no_momentum.step()

        optimizer_with_momentum.zero_grad()
        loss2 = f(x2)
        xs2.append(x2.item())
        ys2.append(loss2.item())
        loss2.backward()
        optimizer_with_momentum.step()

    final_x1 = x1.item()
    final_f_x1 = f(x1).item()
    print(f"SGD without momentum final x: {final_x1:.4f}, f(x) = {final_f_x1:.4f}")

    final_x2 = x2.item()
    final_f_x2 = f(x2).item()
    print(f"SGD with momentum final x: {final_x2:.4f}, f(x) = {final_f_x2:.4f}")

    x_vals = torch.linspace(-3, 2.5, 100)
    y_vals = f(x_vals)

    plt.plot(x_vals, y_vals, color="gray", label="f(x) = x^4 + x^3 - 5x^2")
    plt.plot(xs1, ys1, color="blue", label="SGD (momentum=0.0)")
    plt.scatter(xs1, ys1, color="blue", s=7)
    plt.plot(xs2, ys2, color="orange", label="SGD (momentum=0.9)")
    plt.scatter(xs2, ys2, color="orange", s=7)
    plt.title("SGD trajectory with and without momentum")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    def f2(x, y):
        return 10 * x**2 + 0.1 * y**2

    x = torch.tensor([4.0], requires_grad=True)
    y = torch.tensor([4.0], requires_grad=True)
    optimizer = SGD([x, y], lr=0.1, momentum=0.9)
    xs = []
    ys = []
    for _ in range(num_steps):
        optimizer.zero_grad()
        loss = f2(x, y)
        xs.append(x.item())
        ys.append(y.item())
        loss.backward()
        optimizer.step()

    print(f"Momentum SGD on ravine final: x={x.item():.4f}, y={y.item():.4f}")
    # plot the f function with contour plot
    x_vals = torch.linspace(-5, 5, 100)
    y_vals = torch.linspace(-5, 5, 100)
    X, Y = torch.meshgrid(x_vals, y_vals)
    Z = f2(X, Y)
    plt.contourf(X, Y, Z, levels=30, cmap="Greys")
    plt.plot(xs, ys, color="orange", label="SGD (momentum=0.9)")
    plt.scatter(xs, ys, color="orange", s=7)
    plt.title("SGD with momentum zig-zagging in a ravine")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return 0
