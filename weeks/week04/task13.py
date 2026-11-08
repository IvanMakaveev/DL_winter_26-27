from dl_lib.engine.value import Value


def main() -> int:
    a = Value(2.0, "a")

    print("a + 1.0 =", a + 1.0)
    print("1.0 + a =", 1.0 + a)
    print("a * 2.0 =", a * 2.0)
    print("2.0 * a =", 2.0 * a)
    print("a ** 2.0 =", a**2.0)
    print("a / 2.0 =", a / 2.0)
    print("a.exp() =", a.exp())

    return 0
