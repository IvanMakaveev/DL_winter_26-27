from dl_lib.engine.value import Value


def main() -> int:
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)

    print("a =", a)
    print("b =", b)
    print("c =", c)

    print("a + b =", a + b)
    print("a * b =", a * b)
    print("a * b + c =", a * b + c)

    return 0
