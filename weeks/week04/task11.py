class OverwritingValue:
    def __init__(self, data: float, label: str = ""):
        self.data = data
        self.label = label
        self.grad = 0.0
        self._op = None
        self._prev = set()
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"OverwritingValue(data={self.data})"

    def __add__(self, other):
        if not isinstance(other, OverwritingValue):
            other = OverwritingValue(other)

        result = OverwritingValue(self.data + other.data)
        result._op = "+"
        result._prev = {self, other}

        def _backward():
            self.grad = result.grad
            other.grad = result.grad

        result._backward = _backward
        return result

    def backward(self):

        self.grad = 1.0
        self._backward()
        for child in self._prev:
            child._backward()


def main() -> int:
    a = OverwritingValue(3.0, "a")
    b = a + a

    b.backward()

    print("b.data =", b.data)
    print("a.grad =", a.grad)

    return 0
