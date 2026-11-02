from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import dataset_loss, forward, initialize_weights, gradient_descent_step, analytical_gradient, numerical_gradient

def main() -> int:
    dataset = create_dataset(6)
    weights = initialize_weights(0, 10, 1, 42)
    inputs = [x for x, _ in dataset]
    targets = [y for _, y in dataset]

    print(f"Seed 42, initial w={weights[0]:.6f}")
    for epoch in range(500):
        weight_grad, _ = analytical_gradient(dataset, weights)
        weights = gradient_descent_step(weights, weight_grad, learning_rate=0.001)

    print(f"After 500 epochs, w={weights[0]:.6f}, loss={dataset_loss(dataset, targets, weights):.6f} (target w=2.0)")

    print("Training again without a seed:")
    weights = initialize_weights(0, 10, 1)
    print(f"No seed, initial w={weights[0]:.6f}")
    for epoch in range(500):
        weight_grad, _ = analytical_gradient(dataset, weights)
        weights = gradient_descent_step(weights, weight_grad, learning_rate=0.001)

    print(f"After 500 epochs, w={weights[0]:.6f}, loss={dataset_loss(dataset, targets, weights):.6f} (target w=2.0)")
    return 0