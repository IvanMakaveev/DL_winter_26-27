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
        self.weights = functional.initialize_weights(low, high, input_size, seed)
        self.use_bias = use_bias
        self.bias = (
            functional.initialize_weights(low, high, 1, seed)[0] if use_bias else 0.0
        )
        self.activation = activation

    def forward(self, inputs: np.ndarray) -> float:
        return functional.forward(inputs, self.weights, self.bias, self.activation)
