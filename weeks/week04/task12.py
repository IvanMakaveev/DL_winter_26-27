from dl_lib.engine.value import Value


def main() -> int:
    a = Value(3.0, "a")
    b = a + a

    b.backward()

    print("b.data =", b.data)
    print("a.grad =", a.grad)

    return 0
