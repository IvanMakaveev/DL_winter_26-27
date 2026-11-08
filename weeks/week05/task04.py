from dl_lib.engine.mlp import MLP


def main() -> int:
    x = [2.0, 3.0, -1]
    mlp = MLP(3, [4, 4, 1])

    result = mlp(x)

    print("mlp(x) =", result)
    print("Number of layers:", len(mlp.layers))
    return 0
