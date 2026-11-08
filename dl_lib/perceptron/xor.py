import numpy as np

from dl_lib.activations.functional import sigmoid
from dl_lib.perceptron.functional import initialize_weights


class Xor:
    def __init__(
        self,
        low: float = -1.0,
        high: float = 1.0,
        seed: int | None = None,
    ):
        parameters = initialize_weights(low, high, 9, seed)
        self.hidden_weights = parameters[:4].reshape(2, 2)
        self.hidden_bias = parameters[4:6]
        self.output_weights = parameters[6:8]
        self.output_bias = float(parameters[8])

    def forward(self, inputs: np.ndarray) -> float:
        hidden = sigmoid(np.dot(inputs, self.hidden_weights.T) + self.hidden_bias)
        output = np.dot(hidden, self.output_weights) + self.output_bias
        return float(sigmoid(output))


def forward(model: Xor, x1: float, x2: float) -> float:
    return model.forward(np.array([x1, x2]))
