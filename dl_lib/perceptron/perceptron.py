from collections.abc import Callable
from typing import Any

import numpy as np

from dl_lib.perceptron import functional


class Perceptron:
    def __init__(
        self,
        input_size: int,
        low: float = -1.0,
        high: float = 1.0,
        use_bias: bool = False,
        activation: Callable[..., Any] | None = None,
        seed: int | None = None,
    ):
        self.use_bias = use_bias
        generated_weights = functional.initialize_weights(
            low, high, input_size + (1 if use_bias else 0), seed
        )
        self.weights = generated_weights[:-1] if use_bias else generated_weights
        self.bias = generated_weights[-1] if use_bias else 0.0
        self.activation = activation

    def forward(self, inputs: np.ndarray) -> float:
        return functional.forward(inputs, self.weights, self.bias, self.activation)
