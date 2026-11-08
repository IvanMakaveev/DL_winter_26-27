from dl_lib.engine.graph import draw_dot
from dl_lib.engine.value import Value


def main() -> int:
    x1 = Value(2.0, "x1")
    x2 = Value(0.0, "x2")
    w1 = Value(-3.0, "w1")
    w2 = Value(1.0, "w2")
    b = Value(6.8813735870195432, "b")
    A1 = x1 * w1
    A1.label = "x1*w1"
    A2 = x2 * w2
    A2.label = "x2*w2"

    B1 = A1 + A2
    B1.label = f"{A1.label} + {A2.label}"

    logit = B1 + b
    logit.label = "logit"

    o = logit.tanh()
    o.label = "o"

    o.grad = 1
    logit.grad = 1 - o.data**2

    b.grad = logit.grad
    B1.grad = logit.grad

    A1.grad = B1.grad
    A2.grad = B1.grad

    x1.grad = A1.grad * w1.data
    w1.grad = A1.grad * x1.data

    x2.grad = A2.grad * w2.data
    w2.grad = A2.grad * x2.data

    print("x1.grad =", x1.grad)
    print("w1.grad =", w1.grad)
    print("x2.grad =", x2.grad)
    print("w2.grad =", w2.grad)
    dot = draw_dot(o)
    print(dot.source)

    return 0
