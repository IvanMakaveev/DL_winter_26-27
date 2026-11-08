from dl_lib.activations.functional import sigmoid
from dl_lib.losses.functional import binary_cross_entropy
from dl_lib.perceptron.datasets import nand_dataset
from dl_lib.perceptron.perceptron import Perceptron
from dl_lib.perceptron.training import train


def main() -> int:
    nand_dataset_data = nand_dataset()
    nand_model = Perceptron(
        2, low=-1.0, high=1.0, use_bias=True, activation=sigmoid, seed=42
    )
    nand_history = train(
        nand_model,
        nand_dataset_data,
        learning_rate=0.1,
        epochs=100000,
        loss=binary_cross_entropy,
        log_every=10000,
    )
    for epoch, loss in nand_history:
        print(f"  epoch {epoch:6d}: loss={loss:.6f}")
    print(f"NAND gate - final weights: {nand_model.weights}, bias: {nand_model.bias}")
    for x, y in nand_dataset_data:
        prediction = nand_model.forward(x)
        print(f"  NAND({x[0]}, {x[1]}) = {y} -> predicted {prediction:.6f}")

    return 0
