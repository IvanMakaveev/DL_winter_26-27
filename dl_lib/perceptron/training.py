from collections.abc import Callable

import numpy as np

from dl_lib.losses.functional import mse
from dl_lib.perceptron.functional import (
    dataset_loss,
    dataset_loss_given_bias,
    gradient_descent_step,
    numerical_gradient,
)
from dl_lib.perceptron.perceptron import Perceptron


def train(
    model: Perceptron,
    dataset,
    learning_rate: float,
    epochs: int,
    loss: Callable[..., float] = mse,
    log_every: int = 1,
) -> list[tuple[int, float]]:
    targets = np.array([y for _, y in dataset])
    history = []
    for epoch in range(epochs):
        numerical_grad_weights = numerical_gradient(
            lambda w: dataset_loss(
                dataset, targets, w, model.bias, model.activation, loss
            ),
            model.weights,
        )
        numerical_grad_bias = (
            numerical_gradient(
                lambda b: dataset_loss_given_bias(
                    dataset, targets, model.weights, model.activation, b, loss
                ),
                np.array([model.bias]),
            )
            if model.use_bias == True
            else np.array([0.0])
        )
        model.weights = gradient_descent_step(
            model.weights, numerical_grad_weights, learning_rate
        )
        model.bias = (
            gradient_descent_step(
                np.array([model.bias]), numerical_grad_bias, learning_rate
            )[0]
            if model.use_bias == True
            else 0.0
        )

        if (epoch + 1) % log_every == 0:
            current_loss = dataset_loss(
                dataset, targets, model.weights, model.bias, model.activation, loss
            )
            history.append((epoch + 1, current_loss))
    return history
