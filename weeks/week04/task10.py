from dl_lib.engine.value import Value


def main() -> int:
    a = Value(2.0, "a")
    b = Value(-3, "b")
    c = a + b

    c.backward()

    print("c =", c)
    print("a.grad =", a.grad)
    print("b.grad =", b.grad)

    return 0
