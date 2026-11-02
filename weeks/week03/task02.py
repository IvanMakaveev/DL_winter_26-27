from dl_lib.perceptron import datasets
from dl_lib.perceptron import functional
from dl_lib.losses.functional import mse

import numpy as np

def main() -> int:
    dataset = datasets.create_dataset(6)
    weight = functional.initialize_weights(0, 10, 1, 42)[0]
    print(f"Dataset: {dataset}")
    print(f"w: {weight}")

    inputs = np.array([x for x, _ in dataset])
    targets = np.array([y for _, y in dataset])
    predicted = np.array([functional.forward(np.array([x]), np.array([weight])) for x in inputs])
    mse_loss = mse(predicted, targets)
    print(f"MSE: {mse_loss}")

    print(f"Perturbing w:")
    for delta in [0.002, 0.001, -0.001, -0.002]:
        adjusted_weight = weight + delta
        predicted = np.array([functional.forward(np.array([x]), np.array([adjusted_weight])) for x in inputs])
        mse_loss = mse(predicted, targets)
        print(f"w + ({delta}) -> w={adjusted_weight:.6f}, MSE: {mse_loss:.6f}")

    return 0