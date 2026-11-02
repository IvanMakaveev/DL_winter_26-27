from dl_lib.losses.functional import mse
from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import analytical_gradient, dataset_loss, numerical_gradient, forward, initialize_weights

import numpy as np


def main() -> int:
    dataset = create_dataset(6)
    inputs = [x for x, _ in dataset]
    targets = np.array([y for _, y in dataset])
    weights = initialize_weights(0, 10, 1, 42)

    loss_result = dataset_loss(dataset, targets, weights)
    numerical_grad = numerical_gradient(lambda w: mse(np.array([forward(np.array([x]), np.array([w[0]])) for x in inputs]), targets), weights)
    analytical_grad, _ = analytical_gradient(dataset, weights)
    print(f"Loss: {loss_result}")
    print(f"Numerical gradient: {numerical_grad[0]:.6f}")
    print(f"Analytical gradient: {analytical_grad[0]:.6f}")
    print(f"Difference: {abs(numerical_grad[0] - analytical_grad[0]):.6f}") 
    return 0