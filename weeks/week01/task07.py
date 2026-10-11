import numpy as np


def main() -> int:
    matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    print("Matrix:")
    print(matrix)
    print()

    print("Second row:", matrix[1])
    print("Third column:", matrix[:, 2])
    print("Submatrix:")
    print(matrix[0:2, 1:3])
    print()

    print("Boolean mask for more than 6:")
    print(matrix > 6)
    print()

    print("Values satisfying the mask:", matrix[matrix > 6])

    return 0
