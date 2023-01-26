import typing
import numpy as np
import numpy.ma as ma
from scipy import signal
from edunets.function import Function
from edunets.utils import expand_by_repeating, unpad

# === Unary ops ===

class UnaryOp(Function):
    """
    Unary Operations super class.

    Attributes
    ----------
    + a: TensorOpArgs
        A Tensor (after it goes through self.__prepare__) on which the unary
        operation will be performed. As a Tensor it has both a `data` attribute
        storing the actual result of the operation and a `_upgrade_grad` method
        used to calculate the backward pass
    """
    def __init__(self, a):
        self.a = self.__prepare__(a)
        super().__init__(self.a)


class exp(UnaryOp):
    """
    + Exponential function. 
    The 'derivative' is the result of the operation out = exp(a) 
    """
    op: str = "e"

    def forward(self) -> np.ndarray:
        return np.exp(self.a.data)

    def backward(self) -> None:
        self.a._update_grad(self.out.data * self.out.grad)


class log(UnaryOp):
    """
    + Natural log function. 
    The 'derivative' is the inverse of a 
    """
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
    """
    + Cosinus
    The 'derivative' is -sin of a
    """
    op: str = "cos"

    def forward(self) -> np.ndarray:
        return np.cos(self.a.data)

    def backward(self) -> None:
        self.a._update_grad(-np.sin(self.a.data) * self.out.grad)


class T(UnaryOp):
    """
    + Transposition
    """
    op: str = "T"

    def forward(self) -> np.ndarray:
        return self.a.data.T

    def backward(self) -> None:
        self.a._update_grad(self.out.grad.T)


class expand(Function):
    """
    + Expand a tensor
    """
    op: str = "expand"

    def __init__(self, a, sizes):
        assert isinstance(sizes, tuple), f"Expected tuple for sizes, got {type(sizes)} instead."
        self.sizes = sizes
        self.a = self.__prepare__(a)
        super().__init__(self.a)

    def forward(self) -> np.ndarray:
        return np.broadcast_to(self.a.data, self.sizes)

    def backward(self) -> None:
        def _backward(x) -> np.ndarray:
            # Collapse extra dimensions
            for _ in range(len(self.sizes) - len(self.a.shape)):
                x = x.sum(axis=-1)
            # Sum on remaining dimensions
            for axis, (new_axis_dim, axis_dim) in enumerate(zip(self.a.shape, x.shape)):
                if new_axis_dim != axis_dim:
                    x = x.sum(axis=axis)
            return x.reshape(self.a.shape)

        self.a._update_grad(_backward(self.out.grad))


class reshape(Function):
    """
    + Reshape tensor
    """
    op: str = "reshape"

    def __init__(self, a, shape):
        self.shape = shape
        self.a = self.__prepare__(a)
        super().__init__(self.a)

    def forward(self) -> np.ndarray:
        return self.a.data.reshape(self.shape)

    def backward(self) -> None:
        self.a._update_grad(self.out.grad.reshape(self.a.shape))


class tile(Function):
    op: str = "tile"

    def __init__(self, a, reps):
        self.reps = reps
        self.a = self.__prepare__(a)
        super().__init__(self.a)

    def forward(self) -> np.ndarray:
        return np.tile(self.a.data, self.reps)

    def backward(self) -> None:
        self.a._update_grad((np.prod(self.reps) * self.out.grad)[tuple(slice(i) for i in self.a.shape)])


class pad(Function):
    """
    + Padding
    """
    op: str = "pad"

    def __init__(self, a, pad, mode, **kwargs):
        assert len(pad) % 2 == 0, "Padding length must be divisible by 2."
        assert len(pad) // 2 <= len(a.shape), "Padding length too large."

        # padding must be changed given the difference between numpy convention
        # and pytorch's.
        n = len(a.shape)
        self.pad = tuple(pad[i:i+n] for i in range(0, len(pad), n))[::-1]
        self.mode = mode
        self.kwargs = kwargs

        self.a = self.__prepare__(a)
        super().__init__(self.a)

    def forward(self) -> np.ndarray:
        return np.pad(self.a.data, self.pad, self.mode, **self.kwargs)
    
    def backward(self) -> None:
        self.a._update_grad(unpad(self.out.grad, self.pad))


# === Binary ops ===


class BinaryOp(Function):
    """
    Binary Operations super class.

    Attributes
    ----------
    + a, b: TensorOpArgs, TensorOpArgs
        Two Tensor (after they go through self.__prepare__) on which the binary
        operation will be performed. Both Tensors have a `data` attribute
        storing the actual results of the operations and a `_upgrade_grad` method
        used to calculate the backward pass.
    + brodcastable: bool
        if set to true, tensors will be expanded in order to match the shape of
        the other Tensor it is being computed with.
        Example: [2] * [1, 1, 1] will be changed to [2, 2, 2] * [1, 1, 1] prior
        to the actual multiplication of matrices
    """
    brodcastable: bool = False

    def __init__(self, a, b):
        self.a, self.b = self.__prepare__(a, b)
        super().__init__(self.a, self.b, brodcastable=self.brodcastable)


class add(BinaryOp):
    """
    + Addition.
    The 'derivative' of a + b is 1 (respect to a and respect to b).
    """
    op: str = "+"
    brodcastable: bool = True

    def forward(self) -> np.ndarray:
        return self.a.data + self.b.data

    def backward(self) -> None:
        self.a._update_grad(self.out.grad)
        self.b._update_grad(self.out.grad)


class mul(BinaryOp):
    """
    + Multiplication.
    The 'derivative' of a * b is a (respect to b) and b (respect to a).
    """
    op: str = "*"
    brodcastable: bool = True

    def forward(self) -> np.ndarray:
        return self.a.data * self.b.data

    def backward(self) -> None:
        self.a._update_grad(self.b.data * self.out.grad)
        self.b._update_grad(self.a.data * self.out.grad)


class pow(BinaryOp):
    """
    + Pow
    The 'derivative' of a**c is c*a**(c-1)
    """
    op: str = "**"

    def forward(self) -> np.ndarray:
        return self.a.data ** self.b.data

    def backward(self) -> None:
        self.a._update_grad(self.b.data * (self.a.data ** (self.b.data - 1)) * self.out.grad)

        if len(self.a.shape) == len(self.b.shape):
            self.b._update_grad(self.out.data * np.log(self.a.data) * self.out.grad)
        else:
            self.b._update_grad((self.out.data * np.log(self.a.data) * self.out.grad).sum(axis=0))


class matmul(BinaryOp):
    op: str = "@"

    def forward(self) -> np.ndarray:
        if self.a.data.shape[-1] != self.b.data.shape[0]:
            raise ValueError(f"Invalid matrix operation: {self.a.shape} dot {self.b.shape}.")
        return self.a.data @ self.b.data

    def _backward_a(self) -> np.ndarray:
        if self.a._is_static: return

        if self.out.grad.shape == () or self.out.grad.shape == (1,):
            a_grad = self.b.data.reshape(self.a.shape)
        else:
            a_grad = self.out_grad @ self.b_data.T
            if len(self.a.shape) == 1: a_grad = np.squeeze(a_grad, axis=0)

        return a_grad

    def _backward_b(self) -> np.ndarray:
        if self.b._is_static: return

        if self.out.grad.shape == () or self.out.grad.shape == (1,):
            b_grad = self.a.data.reshape(self.b.shape)
        else:
            b_grad = self.a_data.T @ self.out_grad
            if len(self.b.shape) == 1: b_grad = np.squeeze(b_grad, axis=0)

        return b_grad

    def backward(self) -> None:
        # Return None if both are static
        if self.a._is_static and self.b._is_static: return

        # Convert vectors to correct shape (n,) -> (1,n)
        self.a_data = np.expand_dims(self.a.data, axis=0) if len(self.a.shape) == 1 else self.a.data
        self.b_data = np.expand_dims(self.b.data, axis=0) if len(self.b.shape) == 1 else self.b.data
        self.out_grad = np.expand_dims(self.out.grad, axis=0) if len(self.out.grad.shape) == 1 else self.out.grad

        self.a._update_grad(self._backward_a())
        self.b._update_grad(self._backward_b())


class correlate(Function):
    """
    + Cross-correlation with padding equal to 0 and stride equal to 1.
        This cross-correlation operation is fast thanks to scipy implementation
        of cross-correlation with fft (when fft is assumed to be fasted then regular computation). 
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html)
    """
    op: str = "cross-corr"

    def __init__(self, a, b, method):
        self.method = method
        self.a, self.b = self.__prepare__(a, b)
        super().__init__(self.a, self.b)

    def forward(self) -> np.ndarray:
        if self.a.dim != self.b.dim:
            raise ValueError(f"a (shape: {self.a.shape}) and b (shape: {self.b.shape}) should have the same dimensionality.")
        return signal.correlate(self.a.data, self.b.data, "valid", self.method)

    def backward(self) -> None:
        # Got this thanks to: https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c
        self.a._update_grad(signal.correlate(self.out.grad, self.b.data, "full", self.method))
        self.b._update_grad(signal.correlate(self.a.data, self.out.grad, "valid", self.method))


class cmp(Function):
    def __init__(self, a, b, cmp_fn, cmp_op):
        self.op = cmp_op
        self.dtype = bool
        self.cmp_fn = cmp_fn
        self.a, self.b = self.__prepare__(a, b)
        super().__init__(self.a, self.b)

    def forward(self) -> np.ndarray:
        return self.cmp_fn(self.a.data, self.b.data)

    # No backward for comparisons
    def backward(self) -> None: pass



# == Reduction Ops ===


class ReductionOp(Function):
    """
    Reduction Operations super class.

    Reduction opertations will decrease some dimensions of the Tensor
    by doing some sort of aggregation like sum or max.

    Attributes
    ----------
    + a: TensorOpArgs
        A Tensor (after it goes through self.__prepare__) on which the reduction
        operation will be performed. As a Tensor it has both a `data` attribute
        storing the actual result of the operation and a `_upgrade_grad` method
        used to calculate the backward pass.
    + axis: Tuple[int, ...]
        the dimensions where the reduction will be applied.
    + keepdims: bool
        if set to true the dimensions are unchanged by adding
        a 'singleton' dimension arround the aggregated results
        example: 
            with keepdims: [[1, 2], [1, 3], [1, 4]] -- sum(axis=1) --> [[3], [4], [5]]
            without keepdims: [[1, 2], [1, 3], [1, 4]] -- sum(axis=1) --> [3, 4, 5]
    """
    def __init__(self, a, axis: typing.Tuple[int, ...]=None, keepdims: bool=False):
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
    """
    __getitem__ operation class.

    Attributes
    ----------
    + a: TensorOpArgs
        A Tensor (after it goes through self.__prepare__) on which the getitem
        operation will be performed. As a Tensor it has both a `data` attribute
        storing the actual result of the operation and a `_upgrade_grad` method
        used to calculate the backward pass.
    + items: Union[int, slice, ellipsis, typing.List[int], np.ndarray, Tensor]
        The items that should be extracted from the Tensor
    """
    op: str = "[*]"


    def __prepare_item__(self, item):
        """
        Prepare items so they can act like indexes
        """
        from edunets.tensor import Tensor

        if isinstance(item, Tensor):
            item = item.data if item.dtype == bool else item.data.astype(int)
        if isinstance(item, list):
            item = [it.data if isinstance(it, Tensor) else it for it in item]
        if isinstance(item, np.ndarray):
            item = item if item.dtype == bool else item.astype(int)

        return item


    def __init__(self, a, items):
        """
        Paramaters are explained in this class' docstring
        """
        self.a = a

        if isinstance(items, tuple):
            self.items = tuple(self.__prepare_item__(it) for it in items)
        else:
            self.items = self.__prepare_item__(items)

        super().__init__(self.a)


    def forward(self) -> np.ndarray:
        return self.a.data[self.items]


    def backward(self) -> None:
        z = np.zeros(self.a.shape)
        z[self.items] = self.out.grad
        self.a._update_grad(z)