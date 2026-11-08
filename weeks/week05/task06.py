import numpy as np

from dl_lib.engine.mlp import MLP
from dl_lib.engine.value import Value
from dl_lib.losses.functional import mse


def main() -> int:
    mlp = MLP(3, [4, 4, 1])

    inputs = [[2.0, 3.0, -1], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    targets = np.array([1.0, -1.0, -1.0, 1.0])
    predicted_values = []
    for x in inputs:
        result = mlp(x)
        if not isinstance(result, Value):
            raise TypeError("Expected MLP to produce one output value")
        predicted_values.append(result.data)
    predicted = np.array(predicted_values)
    loss = mse(targets, predicted)

    print("Predictions:", predicted)
    print("MSE loss:", loss)

    return 0
