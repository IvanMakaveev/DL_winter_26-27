from dl_lib.engine.value import Value


def main() -> int:
    decomposed_x1 = Value(2.0, "x1")
    decomposed_x2 = Value(0.0, "x2")
    decomposed_w1 = Value(-3.0, "w1")
    decomposed_w2 = Value(1.0, "w2")
    decomposed_b = Value(6.8813735870195432, "b")
    decomposed_A1 = decomposed_x1 * decomposed_w1
    decomposed_A1.label = "x1*w1"
    decomposed_A2 = decomposed_x2 * decomposed_w2
    decomposed_A2.label = "x2*w2"

    decomposed_B1 = decomposed_A1 + decomposed_A2
    decomposed_B1.label = f"{decomposed_A1.label} + {decomposed_A2.label}"

    decomposed_logit = decomposed_B1 + decomposed_b
    decomposed_logit.label = "logit"

    decomposed_o = (decomposed_logit.exp() + -1 * (-1 * decomposed_logit).exp()) / (
        decomposed_logit.exp() + (-1 * decomposed_logit).exp()
    )
    decomposed_o.label = "o"
    decomposed_o.backward()
    print("o (decomposed) =", decomposed_o)
    print("x1.grad (decomposed) =", decomposed_x1.grad)
    print("w1.grad (decomposed) =", decomposed_w1.grad)

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
    o.backward()
    print("o (built-in tanh) =", o)
    print("x1.grad (built-in tanh) =", x1.grad)
    print("w1.grad (built-in tanh) =", w1.grad)

    return 0
