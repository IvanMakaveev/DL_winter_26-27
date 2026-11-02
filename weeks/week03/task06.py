from dl_lib.perceptron.datasets import and_dataset, or_dataset
from dl_lib.perceptron.functional import initialize_weights
from dl_lib.perceptron.training import train
from dl_lib.perceptron.perceptron import Perceptron
from dl_lib.losses.functional import mse

def main() -> int:
    and_dataset_data = and_dataset()
    and_model = Perceptron(2, low=-1.0, high=1.0, use_bias=False, seed=42)

    print(f"AND gate - initial weights: {and_model.weights}")

    and_history = train(and_model, and_dataset_data, learning_rate=0.1, epochs=100000, loss=mse, log_every=10000)
    for epoch, loss in and_history:
        print(f"  epoch {epoch:6d}: loss={loss:.6f}")
    print(f"AND gate - final weights: {and_model.weights}")
    print("AND gate - predictions:")
    for x, y in and_dataset_data:
        prediction = and_model.forward(x)
        print(f"  AND({x[0]}, {x[1]}) = {y} -> predicted {prediction:.6f}")

    or_dataset_data = or_dataset()
    or_model = Perceptron(2, low=-1.0, high=1.0, use_bias=False, seed=42)
    print(f"OR gate - initial weights: {or_model.weights}")
    or_history = train(or_model, or_dataset_data, learning_rate=0.1, epochs=100000, loss=mse, log_every=10000)
    for epoch, loss in or_history:
        print(f"  epoch {epoch:6d}: loss={loss:.6f}")
    print(f"OR gate - final weights: {or_model.weights}")
    print("OR gate - predictions:")
    for x, y in or_dataset_data:
        prediction = or_model.forward(x)
        print(f"  OR({x[0]}, {x[1]}) = {y} -> predicted {prediction:.6f}")

    return 0