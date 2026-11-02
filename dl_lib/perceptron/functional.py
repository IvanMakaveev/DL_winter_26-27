import numpy as np

def initialize_weights(low: float, high: float, size: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.uniform(low, high, size)

