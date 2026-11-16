import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self.ndim = self.data.ndim
        self.shape = self.data.shape


    def backward(self):
        if self.ndim > 0:
            raise ValueError


    def convert_to_tensor(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return other

    def __add__(self, other: "Tensor"):
        other = self.convert_to_tensor(other)
        return Tensor(self.data + other.data, self.requires_grad if self.requires_grad else other.requires_grad)

    def __sub__(self, other: "Tensor"):
        other = self.convert_to_tensor(other)
        return Tensor(self.data - other.data, self.requires_grad if self.requires_grad else other.requires_grad)

    def __truediv__(self, other: "Tensor"):
        other = self.convert_to_tensor(other)
        return Tensor(self.data / other.data, self.requires_grad if self.requires_grad else other.requires_grad)
    
    def __mul__(self, other: "Tensor"):
        other = self.convert_to_tensor(other)
        return Tensor(self.data * other.data, self.requires_grad if self.requires_grad else other.requires_grad)

    def __neg__(self):
        return Tensor(-self.data, self.requires_grad)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        other = self.convert_to_tensor(other)
        return Tensor(other.data - self.data, self.requires_grad if self.requires_grad else other.requires_grad)

    def __rtruediv__(self, other):
        other = self.convert_to_tensor(other)
        return Tensor(other.data / self.data, self.requires_grad if self.requires_grad else other.requires_grad)