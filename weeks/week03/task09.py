import numpy as np

from dl_lib.activations.functional import sigmoid
from dl_lib.losses.functional import binary_cross_entropy
from dl_lib.perceptron.datasets import and_dataset
from dl_lib.perceptron.functional import (
    analytical_gradient,
    dataset_loss,
    dataset_loss_given_bias,
    numerical_gradient,
)
from dl_lib.perceptron.perceptron import Perceptron


def main() -> int:
    and_dataset_data = and_dataset()
    and_model = Perceptron(
        2, low=-1.0, high=1.0, use_bias=True, activation=sigmoid, seed=42
    )

    print(f"AND gate (sigmoid) - weights: {and_model.weights}, bias: {and_model.bias}")

    numerical_weight_gradient = numerical_gradient(
        lambda w: dataset_loss(
            and_dataset_data,
            np.array([y for _, y in and_dataset_data]),
            w,
            and_model.bias,
            and_model.activation,
            loss=binary_cross_entropy,
        ),
        and_model.weights,
    )
    (analytical_weight_gradient, analytical_bias_gradient) = analytical_gradient(
        and_dataset_data,
        and_model.weights,
        and_model.bias,
        and_model.activation,
        loss="bce",
    )
    print(f"Numerical weight gradient:  {numerical_weight_gradient}")
    print(f"Analytical weight gradient: {analytical_weight_gradient}")
    print(
        f"Weight gradient difference: {analytical_weight_gradient - numerical_weight_gradient}"
    )

    numerical_bias_gradient = numerical_gradient(
        lambda b: dataset_loss_given_bias(
            and_dataset_data,
            np.array([y for _, y in and_dataset_data]),
            and_model.weights,
            and_model.activation,
            b,
            loss=binary_cross_entropy,
        ),
        np.array([and_model.bias]),
    )[0]
    print(f"Numerical bias gradient:  {numerical_bias_gradient:.6f}")
    print(f"Analytical bias gradient: {analytical_bias_gradient:.6f}")
    print(
        f"Bias gradient difference: {analytical_bias_gradient - numerical_bias_gradient:.8f}"
    )

    return 0
