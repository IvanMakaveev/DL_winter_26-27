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
    print("logit =", logit)
    dot = draw_dot(logit)
    print(dot.source)
    return 0
