from dl_lib.engine.graph import draw_dot
from dl_lib.engine.value import Value


def main() -> int:
    a = Value(2.0, "a")
    b = Value(-3.0, "b")
    e = a * b
    e.label = "e"
    c = Value(10.0, "c")
    d = e + c
    d.label = "d"
    f = Value(5.0, "f")
    output = f * d
    output.label = "output"

    print("output =", output)
    print("output.grad =", output.grad)
    print("d.grad =", d.grad)

    dot = draw_dot(output)

    print(dot.source)
    return 0
