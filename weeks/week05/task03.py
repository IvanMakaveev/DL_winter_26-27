from dl_lib.engine.mlp import Layer


def main() -> int:
    x = [2.0, 3.0, -1]
    layer = Layer(3, 4)

    result = layer(x)

    print("layer(x) =", [i.data for i in result])
    print("Number of neurons:", len(result))

    return 0
