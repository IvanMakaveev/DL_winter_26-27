from dl_lib.engine.value import Value


def get_output(a=2.0, b=-3.0, c=10.0, f=5.0):
    e = a * b
    d = e + c
    output = f * d
    return output


def manual_der(output):
    epsilon = 1e-5
    a_grad = (get_output(a=2.0 + epsilon) - get_output(a=2.0 - epsilon)) / (2 * epsilon)
    b_grad = (get_output(b=-3.0 + epsilon) - get_output(b=-3.0 - epsilon)) / (
        2 * epsilon
    )
    c_grad = (get_output(c=10.0 + epsilon) - get_output(c=10.0 - epsilon)) / (
        2 * epsilon
    )
    f_grad = (get_output(f=5.0 + epsilon) - get_output(f=5.0 - epsilon)) / (2 * epsilon)
    return a_grad, b_grad, c_grad, f_grad


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

    a_grad, b_grad, c_grad, f_grad = manual_der(output)
    print(f"Manual gradients: a={a.grad}, b={b.grad}, c={c.grad}, f={f.grad}")
    print(f"Numerical gradients: a={a_grad}, b={b_grad}, c={c_grad}, f={f_grad}")
    return 0
