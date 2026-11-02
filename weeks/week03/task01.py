from dl_lib.perceptron import datasets, functional


def main() -> int:
    dataset = datasets.create_dataset(4)
    print(f"Dataset: {dataset}")
    weights = functional.initialize_weights(0, 100, 1, 42)
    print(f"initialize_weights(0, 100, 1, 42): {weights[0]}")
    weights = functional.initialize_weights(0, 10, 1, 42)
    print(f"initialize_weights(0, 10, 1, 42): {weights[0]}")

    return 0
