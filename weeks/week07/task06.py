import matplotlib.pyplot as plt
import torch

from dl_lib.nn.normalization import BatchNorm1d


def main() -> int:
    torch.manual_seed(42)

    stds = []
    xticks = [2, 8, 64]
    x_fixed = torch.randn(1, 1)
    batchnorm = BatchNorm1d(1)

    for batch_size in xticks:
        results = []
        for _ in range(30):
            others = torch.randn(batch_size - 1, 1)
            batch = torch.cat([x_fixed, others], dim=0)
            with torch.no_grad():
                normalized = batchnorm(batch)
            results.append(normalized[0, 0].item())
        std = torch.tensor(results).std().item()
        print(
            f"Batch size {batch_size}: std of BatchNorm1d(x_fixed) across 30 random batches = {std}"
        )
        stds.append(std)

    plt.plot(xticks, stds)
    plt.scatter(xticks, stds)
    plt.title("BatchNorm1d's output relative to size of batch")
    plt.xlabel("Batch size")
    plt.ylabel("Std of normalized output for the same example")
    plt.tight_layout()
    plt.show()

    return 0
