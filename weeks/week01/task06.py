import numpy as np


def print_array_data(arr: np.ndarray) -> None:
    type = (
        "vector"
        if arr.ndim == 1
        else "matrix"
        if arr.ndim == 2
        else "volume"
        if arr.ndim == 3
        else "hyper-volume"
    )
    print(
        f"{arr.ndim}D array ({type}): shape={arr.shape!s}, dtype={arr.dtype}, ndim={arr.ndim}, size={arr.size}"
    )
    print(arr)


def main() -> int:
    array_1d = np.array([1, 2, 3, 4, 5])
    array_2d = np.array([[1, 2, 3], [4, 5, 6]])
    array_3d = np.array(
        [
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
            [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
        ]
    )

    print_array_data(array_1d)
    print_array_data(array_2d)
    print_array_data(array_3d)

    return 0
