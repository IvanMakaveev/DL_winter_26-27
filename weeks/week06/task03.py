import torch
import matplotlib.pyplot as plt

from dl_lib.optim.optimizers import SGD

def main() -> int:
    def f(x):
        return x**4 + x**3 - 5*x**2

    x = torch.tensor([2.0], requires_grad=True)
    optimizer = SGD([x], lr=0.01)

    trajectory = [x.item()]

    num_steps = 1000
    xs = []
    ys = []
    for _ in range(num_steps):
        optimizer.zero_grad()
        loss = f(x)
        xs.append(x.item())
        ys.append(loss.item())
        loss.backward()
        optimizer.step()

    final_x = x.item()
    final_f_x = f(x).item()
    print(f"Plain SGD final x: {final_x:.4f}, f(x) = {final_f_x:.4f}")

    x_vals = torch.linspace(-3, 2.5, 100)
    y_vals = f(x_vals)

    plt.plot(x_vals, y_vals, label='f(x) = x^4 + x^3 - 5x^2')
    plt.plot(xs, ys, color='red', label='SGD (momentum=0.0)')
    plt.scatter(xs, ys, color='red', s=7)
    plt.title('Plain SGD trajectory trapping in a local minimum')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return 0