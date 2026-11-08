import numpy as np


class Value:
    def __init__(self, data: float, label: str = ""):
        self.data = data
        self.label = label
        self.grad = 0.0
        self._op = None
        self._prev = set()
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        result = Value(self.data + other.data)
        result._op = "+"
        result._prev = {self, other}

        def _backward():
            self.grad += result.grad
            other.grad += result.grad

        result._backward = _backward
        return result

    def __radd__(self, other: float):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        result = Value(self.data * other.data)
        result._op = "*"
        result._prev = {self, other}

        def _backward():
            self.grad += result.grad * other.data
            other.grad += result.grad * self.data

        result._backward = _backward
        return result

    def __rmul__(self, other: float):
        return self * other

    def __pow__(self, other: float):
        result = Value(self.data**other)
        result._op = "**"
        result._prev = {self}

        def _backward():
            self.grad += result.grad * other * (self.data ** (other - 1))

        result._backward = _backward
        return result

    def __truediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        result = Value(self.data / other.data)
        result._op = "/"
        result._prev = {self, other}

        def _backward():
            self.grad += result.grad * (1 / other.data)
            other.grad += result.grad * (-self.data / other.data**2)

        result._backward = _backward
        return result

    def exp(self):
        result = Value(np.exp(self.data))
        result._op = "exp"
        result._prev = {self}

        def _backward():
            self.grad += result.grad * result.data

        result._backward = _backward

        return result

    def tanh(self):

        result = Value(np.tanh(self.data))
        result._op = "tanh"
        result._prev = {self}

        def _backward():
            self.grad += result.grad * (1 - result.data**2)

        result._backward = _backward
        return result

    def backward(self):
        from dl_lib.engine import graph

        self.grad = 1.0
        sorted_nodes = graph.top_sort(self)
        sorted_nodes.reverse()
        for node in sorted_nodes:
            node._backward()
