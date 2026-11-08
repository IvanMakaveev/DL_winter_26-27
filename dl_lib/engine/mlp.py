import random

from dl_lib.engine.value import Value

RNG_DEFAULT = random.Random(42)


class Neuron:
    def __init__(self, n_inputs: int, rng: random.Random = RNG_DEFAULT):
        self.w = [Value(rng.uniform(-1, 1)) for i in range(n_inputs)]
        self.b = Value(rng.uniform(-1, 1))

    def __call__(self, x: list) -> Value:
        logit = sum(x_i * w_i for x_i, w_i in zip(x, self.w)) + self.b
        p_value = logit.tanh()

        return p_value

    def parameters(self) -> list:
        return [*self.w, self.b]


class Layer:
    def __init__(self, n_inputs: int, n_outputs: int, rng: random.Random = RNG_DEFAULT):
        self.neurons = [Neuron(n_inputs, rng) for i in range(n_outputs)]

    def __call__(self, x: list) -> list:
        return [neuron(x) for neuron in self.neurons]

    def parameters(self) -> list:
        result = []
        for neuron in self.neurons:
            result.extend(neuron.parameters())
        return result


class MLP:
    def __init__(
        self, n_inputs: int, n_outputs: list, rng: random.Random = RNG_DEFAULT
    ):
        self.layers = [
            Layer(inputs, outputs, rng)
            for inputs, outputs in zip([n_inputs] + n_outputs[:-1], n_outputs)
        ]

    def __call__(self, x: list) -> list | Value:
        for layer in self.layers:
            x = layer(x)

        return x[0] if len(x) == 1 else x

    def parameters(self) -> list:
        res = []
        for layer in self.layers:
            res.extend(layer.parameters())
        return res
