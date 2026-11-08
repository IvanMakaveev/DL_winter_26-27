import numpy as np

from dl_lib.activations.functional import sigmoid
from dl_lib.perceptron.functional import initialize_weights


class SquareNet:
    def __init__(
        self,
        hidden_units=1,
        low: float = -1.0,
        high: float = 1.0,
        seed: int | None = None,
    ):
        parameters = initialize_weights(low, high, 3 * hidden_units + 1, seed)
        self.hidden_weights = parameters[:hidden_units]
        self.hidden_bias = parameters[hidden_units : 2 * hidden_units]
        self.output_weights = parameters[2 * hidden_units : 3 * hidden_units]
        self.output_bias = float(parameters[3 * hidden_units])

    def forward(self, inputs: np.ndarray) -> float:
        hidden = sigmoid(self.hidden_weights * inputs + self.hidden_bias)
        output = np.dot(hidden, self.output_weights) + self.output_bias
        return float(output)