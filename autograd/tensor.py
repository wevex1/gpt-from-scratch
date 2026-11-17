import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, parents=None, op=None):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self.ndim = self.data.ndim
        self.shape = self.data.shape
        self.parents = parents if parents is not None else ()
        self.op = op


    def backward(self):
        if self.ndim > 0:
            raise ValueError

        self.grad = np.ones_like(self.data)
        order = []
        visited = set()
        for parent in self.parents:
            self.build_order(parent, visited, order)

        order.append(self)

        while order:
            node = order.pop()
            if node.op is None:
                continue

            node.op()


    def build_order(self, node, visited, order):
        if node in visited:
            return

        visited.add(node)
        for parent in node.parents:
            self.build_order(parent, visited, order)

        if node.requires_grad:
            order.append(node)


    def zero_grad(self, node=None, visited=None):
        visited = set() if visited is None else visited

        if node is None:
            node = self

        if node in visited:
            return

        node.grad = None
        visited.add(node)
        for parent in node.parents:
            self.zero_grad(parent, visited)


    def convert_to_tensor(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other


    def accumulate_grad(self, contribution):
        if not self.requires_grad:
            return

        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += contribution

    def __add__(self, other: "Tensor"):
        other = self.convert_to_tensor(other)
        result = Tensor(self.data + other.data)
        result.requires_grad = self.requires_grad if self.requires_grad else other.requires_grad
        result.parents = (self, other)

        def backadd():
            self.accumulate_grad(result.grad)
            other.accumulate_grad(result.grad)

        result.op = backadd
        return result
    

    def __sub__(self, other: "Tensor"):
        other = self.convert_to_tensor(other)
        result = Tensor(self.data - other.data)
        result.requires_grad = self.requires_grad if self.requires_grad else other.requires_grad
        result.parents = (self, other)

        def backsub():
            self.accumulate_grad(result.grad)
            other.accumulate_grad(-result.grad)

        result.op = backsub
        return result


    def __truediv__(self, other: "Tensor"):
        other = self.convert_to_tensor(other)
        result = Tensor(self.data / other.data)
        result.requires_grad = self.requires_grad if self.requires_grad else other.requires_grad
        result.parents = (self, other)

        def backdiv():
            self.accumulate_grad(other.data**-1 * result.grad)
            other.accumulate_grad((-self.data/other.data**2) * result.grad)

        result.op = backdiv
        return result

    
    def __mul__(self, other: "Tensor"):
        other = self.convert_to_tensor(other)
        result = Tensor(self.data * other.data)
        result.requires_grad = self.requires_grad if self.requires_grad else other.requires_grad
        result.parents = (self, other)

        def backmul():
            self.accumulate_grad(other.data * result.grad)
            other.accumulate_grad(self.data * result.grad)

        result.op = backmul
        return result


    def __pow__(self, other):
        result = Tensor(self.data**other)
        result.requires_grad = self.requires_grad
        result.parents = (self,)

        def backpow():
            self.accumulate_grad((other * (self.data**(other-1)) * result.grad))

        result.op = backpow
        return result


    def __neg__(self):
        result = Tensor(-self.data, self.requires_grad)
        result.parents = (self,)

        def backneg():
            self.accumulate_grad(-result.grad)

        result.op = backneg

        return result


    def __radd__(self, other):
        return self + other


    def __rmul__(self, other):
        return self * other


    def __rsub__(self, other):
        other = self.convert_to_tensor(other)
        result = Tensor(other.data - self.data)
        result.requires_grad = self.requires_grad if self.requires_grad else other.requires_grad
        result.parents = (self, other)

        def backsub():
            other.accumulate_grad(result.grad)
            self.accumulate_grad(-result.grad)

        result.op = backsub
        return result


    def __rtruediv__(self, other):
        other = self.convert_to_tensor(other)
        result = Tensor(other.data / self.data)
        result.requires_grad = self.requires_grad if self.requires_grad else other.requires_grad
        result.parents = (self, other)

        def backdiv():
            other.accumulate_grad(self.data**-1 * result.grad)
            self.accumulate_grad((-other.data/self.data**2) * result.grad)

        result.op = backdiv
        return result