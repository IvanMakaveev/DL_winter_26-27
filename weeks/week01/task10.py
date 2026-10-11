import numpy as np


def main() -> int:
    A = np.array([[1, 2], [3, 4]])
    print("A:")
    print(A)
    print()

    B = np.array([[5, 6], [7, 8]])
    print("B:")
    print(B)
    print()

    elementwise_product = A * B
    print("Elementwise:")
    print(elementwise_product)
    print()

    matrix_product = A @ B
    print("Matrix multiplication:")
    print(matrix_product)
    print()

    transpose_A = A.T
    print("The transpose of A:")
    print(transpose_A)

    return 0
