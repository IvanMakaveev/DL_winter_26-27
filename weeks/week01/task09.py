import numpy as np


def main() -> int:
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    matrix_plus_scalar = matrix + 10
    print("Matrix + scalar (10):")
    print(matrix_plus_scalar)
    print()

    row_vector = np.array([10, 20, 30])
    matrix_plus_row_vector = matrix + row_vector
    print("Matrix (2, 3) + row vector (3,):")
    print(matrix_plus_row_vector)
    print()

    column_vector = np.array([[100], [200]])
    matrix_plus_column_vector = matrix + column_vector
    print("Matrix (2, 3) + column vector (2, 1):")
    print(matrix_plus_column_vector)
    print()

    incompatible_vector = np.array([1, 2])
    try:
        matrix + incompatible_vector
    except ValueError as e:
        print("Matrix (2, 3) + incompatible (2,) raises:")
        print(e)

    return 0
