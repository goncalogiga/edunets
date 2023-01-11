import typing
import numpy as np
import numpy.ma as ma
from edunets.expanders import expand_by_repeating
from edunets.tensor import Function, TensorContent, to_tensor, Tensor


# === Unary ops ===

class UnaryOp(Function):
    def __init__(self, a: TensorContent):
        self.a = self.__prepare__(a)
        super().__init__(self.a)


class exp(UnaryOp):
    op: str = "e"

    def forward(self) -> np.ndarray:
        return np.exp(self.a.data)

    def backward(self) -> None:
        self.a._update_grad(self.out.data * self.out.grad)


class log(UnaryOp):
    op: str = "log"

    def forward(self) -> np.ndarray:
        return np.log(self.a.data)

    def backward(self) -> None:
        self.a._update_grad((self.a.data ** (-1)) * self.out.grad)


class relu(UnaryOp):
    op: str = "relu"

    def forward(self) -> np.ndarray:
        return (self.a.data > np.zeros(self.a.shape)) * self.a.data

    def backward(self) -> None:
        self.a._update_grad(self.out.grad)


class cos(UnaryOp):
    op: str = "cos"

    def forward(self) -> np.ndarray:
        return np.cos(self.a.data)

    def backward(self) -> None:
        self.a._update_grad(-np.sin(self.a.data) * self.out.grad)


class sin(UnaryOp):
    op: str = "sin"

    def forward(self) -> np.ndarray:
        return np.sin(self.a.data)

    def backward(self) -> None:
        self.a._update_grad(np.cos(self.a.data) * self.out.grad)


# === Binary ops ===


class BinaryOp(Function):
    brodcastable: bool = False

    def __init__(self, a, b):
        self.a, self.b = self.__prepare__(a, b)
        super().__init__(self.a, self.b, brodcastable=self.brodcastable)


class add(BinaryOp):
    op: str = "+"
    brodcastable: bool = True

    def forward(self) -> np.ndarray:
        return self.a.data + self.b.data

    def backward(self) -> None:
        self.a._update_grad(self.out.grad)
        self.b._update_grad(self.out.grad)


class mul(BinaryOp):
    op: str = "*"
    brodcastable: bool = True

    def forward(self) -> np.ndarray:
        return self.a.data * self.b.data

    def backward(self) -> None:
        self.a._update_grad(self.b.data * self.out.grad)
        self.b._update_grad(self.a.data * self.out.grad)


class pow(BinaryOp):
    op: str = "**"

    def forward(self) -> np.ndarray:
        return self.a.data ** self.b.data

    def backward(self) -> None:
        self.a._update_grad(self.b.data * (self.a.data ** (self.b.data - 1)) * self.out.grad)
        self.b._update_grad(np.zeros(self.b.shape))


class matmul(BinaryOp):
    op: str = "@"

    def forward(self) -> np.ndarray:
        if self.a.data.shape[-1] != self.b.data.shape[0]:
            raise ValueError(f"Invalid matrix operation: {self.a.shape} dot {self.b.shape}.")
        return self.a.data @ self.b.data

    def backward(self) -> None:
        self.a._update_grad(self.out.grad @ self.b.data.T)
        self.b._update_grad(self.a.data.T @ self.out.grad)


# == Reduction Ops ===


class ReductionOp(Function):
    def __init__(self, a: TensorContent, axis: typing.Tuple[int, ...]=None, keepdims: bool=False):
        self.a = self.__prepare__(a)
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(self.a)

    
class max(ReductionOp):
    op: str = "max"

    def forward(self) -> np.ndarray:
        self.mask = ma.masked_equal(self.a.data, np.max(self.a.data))
        return self.mask.fill_value

    def backward(self) -> None:
        self.a._update_grad((self.mask.mask / ma.count_masked(self.mask)) * self.out.grad)


class min(max):
    op: str = "min"

    def forward(self) -> np.ndarray:
        self.mask = ma.masked_equal(self.a.data, np.min(self.a.data))
        return self.mask.fill_value


class sum(ReductionOp):
    op: str = "sum"

    def forward(self) -> np.ndarray:
        return np.sum(self.a.data, axis=self.axis, keepdims=self.keepdims)

    def backward(self) -> None:
        if self.axis:
            self.a._update_grad(expand_by_repeating(self.out.grad, self.a))
        else:
            self.a._update_grad(np.ones(self.a.shape) * self.out.grad)

# === Others ===

class getitem(Function):
    op: str = "[*]"

    def __prepare__(self):
        is_itter: bool = True

        if isinstance(self.items, tuple): 
            self.items = list(self.items)

        if isinstance(self.items, list):
            for i, item in enumerate(self.items):
                if self.is_tensor(item):
                    self.items[i] = item.data.astype(int)
        elif self.is_tensor(self.items):
            self.items = self.items.data.astype(int)
        else:
            is_itter = False

        if is_itter and not isinstance(self.items, slice): 
            self.items = tuple(self.items)

    def __init__(self, a: TensorContent, items: typing.Union[Tensor, TensorContent, slice]):
        self.a = a
        self.items = items

        self.__prepare__()
        super().__init__(self.a)

    def forward(self) -> np.ndarray:
        return self.a.data[self.items]

    def backward(self) -> None:
        z = np.zeros(self.a.shape)
        z[self.items] = self.out.grad
        self.a._update_grad(z)


# === functions without grad ===

def argmax(t: Tensor, **kargs) -> Tensor: return to_tensor(np.argmax(t.data, **kargs))
