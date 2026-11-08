from dl_lib.engine.value import Value


def get_output(a=2.0, b=-3.0, c=10.0, f=5.0):
    e = a * b
    d = e + c
    output = f * d
    return output


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

    output.grad = 1

    d.grad = f.data
    f.grad = d.data

    e.grad = d.grad
    c.grad = d.grad

    a.grad = e.grad * b.data
    b.grad = e.grad * a.data

    learning_rate = 0.01
    new_output = get_output(
        a.data - learning_rate * a.grad,
        b.data - learning_rate * b.grad,
        c.data - learning_rate * c.grad,
        f.data - learning_rate * f.grad,
    )
    print(f"L before step: {output.data}")
    print(f"L after step: {new_output}")
    return 0
