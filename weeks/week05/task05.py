from dl_lib.engine.mlp import MLP


def main() -> int:
    mlp = MLP(3, [4, 4, 1])
    parameters = mlp.parameters()
    print("Number of parameters:", len(parameters))
    print("First three parameters:", [i.data for i in parameters[:3]])

    return 0
