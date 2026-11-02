import numpy as np

def initialize_weights(low: float, high: float, size: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.uniform(low, high, size)

def forward(inputs: np.ndarray, weights: np.ndarray, bias:float = 0.0, activation: callable = None) -> float:
    result = np.dot(inputs, weights) + bias
    if activation:
        result = activation(result)
    return result