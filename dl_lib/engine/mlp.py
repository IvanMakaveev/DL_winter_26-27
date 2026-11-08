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
