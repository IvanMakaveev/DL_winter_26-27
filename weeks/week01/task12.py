import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    x = np.linspace(0, 2 * np.pi, 200)
    y_sin = np.sin(x)
    y_cos = np.cos(x)
    samples = np.random.normal(loc=0, scale=1, size=1000)

    print(f"sin(x) range: [{y_sin.min():.4f}, {y_sin.max():.4f}]")
    print(f"Histogram sample count: {len(samples)}")

    _, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(x, y_sin, label="sin(x)")
    axs[0].plot(x, y_cos, label="cos(x)")
    axs[0].set_title("Line Plot")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].legend()
    axs[0].grid()

    axs[1].hist(samples, bins=32)
    axs[1].set_title("Histogram")
    axs[1].set_xlabel("value")
    axs[1].set_ylabel("count")
    axs[1].grid()

    plt.tight_layout()
    plt.show()

    return 0
