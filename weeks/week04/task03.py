from dl_lib.engine.graph import draw_dot, trace
from dl_lib.engine.value import Value


def main() -> int:
    a = Value(-3.0, "a")
    b = Value(2.0, "b")
    c = a * b
    c.label = "a*b"
    d = Value(10.0, "c")
    e = d + c
    e.label = "c+d"

    nodes, edges = trace(e)
    print("Number of nodes:", len(nodes))
    print("Number of edges:", len(edges))

    dot = draw_dot(e)

    print(dot.source)
    return 0
