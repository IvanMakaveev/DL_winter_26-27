import time

import numpy as np


def time_function(func, *args, **kwargs) -> tuple[int, float]:
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return result, elapsed_time


def sum_loop(arr, size) -> float:
    result = 0.0
    for i in range(size):
        result += arr[i]
    return result


def main() -> int:
    size = 10_000_000
    arr = np.random.rand(size)

    loop_result, loop_time = time_function(sum_loop, arr, size)

    numpy_result, numpy_time = time_function(np.sum, arr)

    print(f"Array length: {size:,}")
    print(f"Loop sum: {loop_result:.4f} ({loop_time:.4f}s)")
    print(f"Vectorized sum: {numpy_result:.4f} ({numpy_time:.4f}s)")
    speedup = loop_time / numpy_time if numpy_time > 0 else float("inf")
    print(f"Vectorized speedup: {speedup:.0f}x")
    print()

    print(f"Vectorized mean: {np.mean(arr):.4f}")
    print(f"Vectorized std: {np.std(arr):.4f}")
    print(f"Vectorized min/max: {np.min(arr):.4f} / {np.max(arr):.4f}")

    return 0
