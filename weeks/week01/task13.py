import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    uniform_samples = np.random.uniform(low=0, high=1, size=10000)
    normal_samples = np.random.normal(loc=0, scale=1, size=10000)
    exponential_samples = np.random.exponential(scale=1, size=10000)

    print(
        f"Uniform: mean={uniform_samples.mean():.4f}, std={uniform_samples.std():.4f}, min={uniform_samples.min():.4f}, max={uniform_samples.max():.4f}"
    )
    print(
        f"Normal: mean={normal_samples.mean():.4f}, std={normal_samples.std():.4f}, min={normal_samples.min():.4f}, max={normal_samples.max():.4f}"
    )
    print(
        f"Exponential: mean={exponential_samples.mean():.4f}, std={exponential_samples.std():.4f}, min={exponential_samples.min():.4f}, max={exponential_samples.max():.4f}"
    )

    _, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].hist(uniform_samples, bins=32)
    axs[0].set_title("Uniform")
    axs[0].set_xlabel("value")
    axs[0].set_ylabel("count")
    axs[0].grid()

    axs[1].hist(normal_samples, bins=32)
    axs[1].set_title("Normal")
    axs[1].set_xlabel("value")
    axs[1].set_ylabel("count")
    axs[1].grid()

    axs[2].hist(exponential_samples, bins=32)
    axs[2].set_title("Exponential")
    axs[2].set_xlabel("value")
    axs[2].set_ylabel("count")
    axs[2].grid()

    plt.tight_layout()
    plt.show()

    return 0
