import numpy as np
import torch

from dl_lib.activations import functional


def main() -> int:
    scores = np.array([1.0, 2.0, 3.0])
    softmax_values = functional.softmax(scores)
    print(f"scores: {scores}")
    print(f"softmax: {softmax_values}")
    print(f"sum of probabilities: {softmax_values.sum():.4f}")
    print()

    large_scores = np.array([1000.0, 1001.0, 1002.0])
    naive_softmax = np.exp(large_scores) / np.sum(np.exp(large_scores))
    stable_softmax = functional.softmax(large_scores)

    print(f"Large scores: {large_scores}")
    print(f"Naive softmax (unstable): {naive_softmax}")
    print(f"Stable softmax: {stable_softmax}")
    print()

    torch_softmax = torch.softmax(torch.from_numpy(scores), dim=0).numpy()

    print("Using PyTorch:")
    print(f"torch softmax: {torch_softmax}")

    return 0
