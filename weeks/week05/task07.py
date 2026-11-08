from dl_lib.engine.mlp import MLP
from dl_lib.engine.value import Value


def predict(inputs: list[list], mlp: MLP) -> list[Value]:
    predicted_values = []
    for x in inputs:
        result = mlp(x)
        if not isinstance(result, Value):
            raise TypeError("Expected MLP to produce one output value")
        predicted_values.append(result)
    return predicted_values


def rss(predicted: list[Value], targets: list[float]) -> Value:
    return sum(((p + (-t)) ** 2 for p, t in zip(predicted, targets)), start=Value(0.0))


def main() -> int:
    mlp = MLP(3, [4, 4, 1])

    inputs = [[2.0, 3.0, -1], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    targets = [1.0, -1.0, -1.0, 1.0]
    learning_rate = 0.05

    for epoch in range(20):
        predicted = predict(inputs, mlp)
        loss = rss(predicted, targets)
        loss.backward()
        for param in mlp.parameters():
            param.data -= learning_rate * param.grad
        print(
            f"epoch {epoch}: loss = {loss.data}, sum|grad| = {sum(abs(param.grad) for param in mlp.parameters())}"
        )

    return 0
