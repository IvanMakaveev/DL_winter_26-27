from dl_lib.engine.mlp import Neuron


def main() -> int:
    inputs = [2.0, 3.0, -1]
    neuron = Neuron(3)

    print("neuron.w =", [i.data for i in neuron.w])
    print("neuron.b =", neuron.b.data)
    print("neuron(x) =", neuron(inputs))

    return 0
