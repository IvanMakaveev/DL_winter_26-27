import matplotlib.pyplot as plt
import torch

from dl_lib.nn.normalization import LayerNorm


def main() -> int:
    lnorm = LayerNorm(4)
    x = torch.tensor(
        [
            [4.0820, 0.4131, -3.3576, 2.1369],
            [-1.1690, -1.7972, 1.8067, 2.6761],
            [-0.4385, 0.1933, -0.1933, 1.3641],
        ]
    )
    print(f"x = {x}")
    lnorm.train()
    training = lnorm(x)
    lnorm.eval()
    evaluating = lnorm(x)
    print(f"Training:   layer_norm(x) = {training}")
    print(f"Evaluating:   layer_norm(x) = {evaluating}")
    print("Identical regardless of mode:", torch.allclose(training, evaluating))

    torch.manual_seed(42)

    stds = []
    xticks = [2, 8, 64]
    x_fixed = torch.randn(1, 1)
    layernorm = LayerNorm(1)

    for batch_size in xticks:
        results = []
        for _ in range(30):
            others = torch.randn(batch_size - 1, 1)
            batch = torch.cat([x_fixed, others], dim=0)
            with torch.no_grad():
                normalized = layernorm(batch)
            results.append(normalized[0, 0].item())
        std = torch.tensor(results).std().item()
        print(
            f"Batch size {batch_size}: std of LayerNorm(x_fixed) across 30 random batches = {std}"
        )
        stds.append(std)

    plt.plot(xticks, stds)
    plt.scatter(xticks, stds)
    plt.title("LayerNorm's output relative to size of batch")
    plt.xlabel("Batch size")
    plt.ylabel("Std of normalized output for the same example")
    plt.tight_layout()
    plt.show()

    return 0
