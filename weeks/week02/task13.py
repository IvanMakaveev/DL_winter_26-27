import numpy as np

from dl_lib.losses import functional


def main() -> int:
    y_true_binary = np.array([1.0, 0.0, 1.0])
    p_binary = np.array([0.9, 0.2, 0.6])
    binary_loss = functional.binary_cross_entropy(y_true_binary, p_binary)
    print(
        f"Binary cross-entropy on y_true={y_true_binary}, p={p_binary}: {binary_loss:.4f}"
    )
    print()

    confidently_right = functional.binary_cross_entropy(
        np.array([1.0]), np.array([0.999])
    )
    confidently_wrong = functional.binary_cross_entropy(
        np.array([1.0]), np.array([0.001])
    )
    print(f"y_true=1, p=0.999 (confidently right): loss={confidently_right:.4f}")
    print(f"y_true=1, p=0.001 (confidently wrong): loss={confidently_wrong:.4f}")
    print()

    y_true_categorical = np.array([0, 1, 2])
    p_categorical = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    categorical_loss = functional.categorical_cross_entropy(
        y_true_categorical, p_categorical
    )
    print(
        f"Multiclass cross-entropy on y_true={y_true_categorical}: {categorical_loss:.4f}"
    )
    print()

    confidently_right_cat = functional.categorical_cross_entropy(
        np.array([0]), np.array([[0.999, 0.001, 0.000]])
    )
    confidently_wrong_cat = functional.categorical_cross_entropy(
        np.array([0]), np.array([[0.001, 0.999, 0.000]])
    )
    print(
        f"y_true=0, p_true=0.999 (confidently right): loss={confidently_right_cat:.4f}"
    )
    print(
        f"y_true=0, p_true=0.001 (confidently wrong): loss={confidently_wrong_cat:.4f}"
    )

    return 0
