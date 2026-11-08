import matplotlib.pyplot as plt

from dl_lib.activations.functional import sigmoid
from dl_lib.losses.functional import mse
from dl_lib.perceptron.datasets import and_dataset
from dl_lib.perceptron.perceptron import Perceptron
from dl_lib.perceptron.training import train


def main() -> int:
    and_dataset_data = and_dataset()
    and_model = Perceptron(2, low=-1.0, high=1.0, use_bias=True, seed=42)

    and_history = train(
        and_model,
        and_dataset_data,
        learning_rate=0.1,
        epochs=100000,
        loss=mse,
        log_every=5,
    )
    print(f"Without sigmoid - final loss: {and_history[-1][1]:.6f}")
    for x, y in and_dataset_data:
        prediction = and_model.forward(x)
        print(f"  AND({x[0]}, {x[1]}) = {y} -> predicted {prediction:.6f}")

    and_model_sigmoid = Perceptron(
        2, low=-1.0, high=1.0, use_bias=True, activation=sigmoid, seed=42
    )

    and_history_sigmoid = train(
        and_model_sigmoid,
        and_dataset_data,
        learning_rate=0.1,
        epochs=100000,
        loss=mse,
        log_every=5,
    )
    print(f"With sigmoid - final loss: {and_history_sigmoid[-1][1]:.6f}")
    for x, y in and_dataset_data:
        prediction = and_model_sigmoid.forward(x)
        print(f"  AND({x[0]}, {x[1]}) = {y} -> predicted {prediction:.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(
        [epoch for epoch, _ in and_history],
        [loss for _, loss in and_history],
        label="Without Sigmoid",
    )
    plt.plot(
        [epoch for epoch, _ in and_history_sigmoid],
        [loss for _, loss in and_history_sigmoid],
        label="With Sigmoid",
    )
    plt.xlabel("epochs")
    plt.ylabel("MSE loss")
    plt.title("AND Gate - loss over 100,000 epochs")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return 0
