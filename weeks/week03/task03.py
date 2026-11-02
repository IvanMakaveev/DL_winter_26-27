from dl_lib.perceptron import datasets
from dl_lib.perceptron import functional
from dl_lib.losses.functional import mse

import numpy as np

def main() -> int:
    dataset = datasets.create_dataset(6)
    weights = functional.initialize_weights(0, 10, 1, 42)
    inputs = np.array([x for x, _ in dataset])
    targets = np.array([y for _, y in dataset])

    def loss_function(w: np.ndarray) -> float:
        predicted = np.array([functional.forward(np.array([x]), np.array([w[0]])) for x in inputs])
        return mse(predicted, targets)

    print(f"Loss before update: {loss_function(weights)}")
    numerical_grad = functional.numerical_gradient(loss_function, weights)
    print(f"Numerical gradient: {numerical_grad[0]}")
    print(f"w after subtracting raw gradient: w={(weights - numerical_grad)[0]}, loss={loss_function(weights - numerical_grad)}")
    
    new_weights = functional.gradient_descent_step(weights, numerical_grad, learning_rate=0.001)
    print(f"w after one step with learning_rate=0.001: w={new_weights[0]}, loss={loss_function(new_weights)}")

    print(f"Loss before update: {loss_function(weights)}")
    for epoch in range(10):
        numerical_grad = functional.numerical_gradient(loss_function, weights)
        weights = functional.gradient_descent_step(weights, numerical_grad, learning_rate=0.001)
        print(f"Epoch {epoch + 1}: w={weights[0]}, loss={loss_function(weights)}")
    
    return 0