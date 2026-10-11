import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    x = np.linspace(0, 10, 50)
    noise = np.random.normal(0, 1, x.shape)
    y = 2.0 * x + 1.0 + noise

    print(f"Generated {len(x)} points for y = 2.0*x + 1.0 + noise")
    print(f"x range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"y range: [{y.min():.2f}, {y.max():.2f}]")

    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Synthetic linear data with noise")
    plt.show()

    return 0
