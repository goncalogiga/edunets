import pprint
import random
import warnings
import numpy as np
import numpy.ma as ma
from edunets.visualisations import draw_dot


class Tensor:
    _op, _children, _is_leaf = "", (), False
    grad = None

    def __init__(self, data, dtype=np.float32, requires_grad=False):
        self.dtype = dtype
        self._backward = lambda: None
        self._key = random.randint(0, 2**32)
        self.data = np.array(data, dtype=dtype)
        self.requires_grad = bool(requires_grad)
        if self.data.shape == (): self.data = np.expand_dims(self.data, axis=0)


    def __repr__(self):
        if self.data.shape == (1,):
            data_repr = pprint.pformat(self.data)[7:-1].replace(']', '')
        else:
            data_repr = pprint.pformat(self.data)[6:-1]

        if self.requires_grad:
            return f"tensor({data_repr}, requires_grad=True)"
        else:
            return f"tensor({data_repr})"


    def _update_grad(self, value):
        if self._is_leaf: return self
        if self.grad is None: self.grad = 0
        self.grad += value 

    
    def _free_grad(self):
        if self.requires_grad is False: self.grad = None


    def _parent_of(self, children):
        self._children = children
        return self

    
    def _result_of_op(self, op):
        self._op = op
        return self


    def _set_as_leaf(self):
        self._is_leaf = True
        return self


    def _set_backward(self, backward):
        self._backward = backward
        return self

    
    def backward(self):
        """
        Backpropagation algorithm
        """
        if self.shape != (1,):
            raise RuntimeError("Edunets' back propagation only supports scalar outputs.")

        sorted_grah, visited = [], set()
        
        def topological_sort(v):
            if v._key not in visited:
                visited.add(v._key)

                for child in v._children:
                    topological_sort(child)

                sorted_grah.append(v)

        # Topological sort of the graph of operations
        topological_sort(self)

        # Initial gradient (last node) is set to 1.0 so we can calculate the first
        # chain rule result.
        # Backpropagation is done assuming the node calling .backward()
        # is the last of the operation grah
        self.grad, prev_v = 1.0, None

        # For each node, apply the chain rule to get its gradient
        for v in reversed(sorted_grah):
            if prev_v: prev_v._free_grad()
            v._backward()
            prev_v = v

    
    def assign(self, x):
        print("assign")
        if not isinstance(x, Tensor):
            x = Tensor(x)
        if self.shape != x.shape:
            raise RuntimeError(f"Expected shape {self.shape}, but got {x.shape} instead.")
        self.data = x.data
        return x


    # === comparisons ===
    def __gt__(self, other): return Tensor(self.data > other.data)
    def __lt__(self, other): return Tensor(self.data < other.data)
    def __ge__(self, other): return Tensor(self.data >= other.data)
    def __le__(self, other): return Tensor(self.data <= other.data)
    def __eq__(self, other): return Tensor(self.data == other.data)
    def __ne__(self, other): return Tensor(self.data != other.data)

    # === base operations ===
    def __getitem__(self, items): return Tensor(self.data[items])
    def __matmul__(self, other): return tmatmul(self, other)
    def __add__(self, other): return tadd(self, other)
    def __mul__(self, other): return tmul(self, other)
    def __pow__(self, other): return tpow(self, other)
    
    def cos(self): return tcossin(self, cos=True)
    def sin(self): return tcossin(self, cos=False)
    def exp(self): return texp(self)
    def log(self): return tlog(self)

    def sum(self): return tsum(self)
    def max(self): return tmaxmin(self, max=True)
    def min(self): return tmaxmin(self, max=False)

    def relu(self): return trelu(self)

    # === operations based on the previous ones ===
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __neg__(self): return -1 * self
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return -self + other
    def __truediv__(self, other): return self * other**(-1)
    def __rtruediv__(self, other): return other * self**(-1)
    def __rpow__(self, other): return other**self
    def matmul(self, other): return self @ other
    def tan(self): return self.sin()/self.cos()

    # === more advanced operations ===
    def mean(self): return self.sum()/sum(x for x in self.shape)

    #   ~~~ activation functions ~~~
    def sigmoid(self): return 1.0/(1.0 + (-self).exp())
    def tanh(self): return 2.0 * ((2.0 * self).sigmoid()) - 1.0

    # === tensor class methods ===
    @classmethod
    def zeros_like(self, t, **kwargs): return self(np.zeros_like(t.data), **kwargs)

    @classmethod
    def zeros(self, *shape, **kwargs): return self(np.zeros(shape), **kwargs)

    @classmethod
    def ones(self, shape, **kwargs): return self(np.ones(shape), **kwargs)

    @classmethod
    def empty(self, *shape, **kwargs): return self(np.empty(shape), **kwargs)

    @classmethod
    def randn(self, *shape, **kwargs): 
        return self(np.random.default_rng().standard_normal(size=shape), **kwargs)

    @classmethod
    def arange(self, stop, start=0, **kwargs): 
        return self(np.arange(start=start, stop=stop), **kwargs)

    @classmethod
    def uniform(self, *shape, **kwargs): 
        return self(np.random.default_rng().random(size=shape) * 2 - 1, **kwargs)

    @classmethod
    def eye(self, dim, **kwargs): return self(np.eye(dim), **kwargs)

    # === tensor propreties ===
    @property
    def graph(self): return draw_dot(self)

    @property
    def shape(self): return self.data.shape

    @property
    def T(self): self.data = self.data.T; return self 


# **** Tensor base operations helper functions ****

def op_wrap(op):
    """
    Decorator to convert constants to Tensor constants
    """
    def wrapper(*args, **kwargs):
        args = [
            arg if isinstance(arg, Tensor) else Tensor(arg, requires_grad=False)._set_as_leaf()\
            for arg in args  
        ]
        return op(*args, **kwargs)
    return wrapper


def op_brodcast(a, b):
    if a.shape == b.shape or a._is_leaf or b._is_leaf:
        return

    try:
        brodcast_shape = np.broadcast(a.data, b.data).shape
    except ValueError:
        raise ValueError(f"Shape of tensors mismatch: {a.shape} x {b.shape}.")
    
    a_shape, b_shape = a.shape, b.shape
    
    a.data = np.broadcast_to(a.data, brodcast_shape)
    b.data = np.broadcast_to(b.data, brodcast_shape)
    
    if (a.requires_grad and a_shape != a.shape) or (b.requires_grad and b_shape != b.shape):
        warnings.warn("""Edunets can only brodcast by reshaping tensors inplace,
        beware of those changes if the brodcasted tensors are used elsewhere.""")


# **** Tensor base operation functions ****


@op_wrap
def tadd(a, b):
    op_brodcast(a, b)

    f = Tensor(a.data + b.data)._parent_of((a, b))._result_of_op('+')

    def backward():
        a._update_grad(f.grad)
        b._update_grad(f.grad)

    return f._set_backward(backward)


@op_wrap
def tmul(a, b):
    op_brodcast(a, b)

    f = Tensor(a.data * b.data)._parent_of((a, b))._result_of_op('*')

    def backward():
        a._update_grad(b.data * f.grad)
        b._update_grad(a.data * f.grad)

    return f._set_backward(backward)


@op_wrap
def texp(a):
    f = Tensor(np.exp(a.data))._parent_of((a, ))._result_of_op('e')

    def backward():
        a._update_grad(f.data * f.grad)

    return f._set_backward(backward)


@op_wrap
def tlog(a):
    f = Tensor(np.log(a))._parent_of((a,))._result_of_op('log')

    def backward():
        a._update_grad(a.data**(-1) * f.grad)

    return f._set_backward(backward)


@op_wrap
def tpow(a, b):
    f = Tensor(a.data ** b.data)._parent_of((a, b))._result_of_op('**')

    def backward():
        a._update_grad(b.data * (a.data**(b.data - 1)) * f.grad)
        b._update_grad(np.zeros(b.shape))

    return f._set_backward(backward)


@op_wrap
def tmatmul(a, b):
    f = Tensor(np.matmul(a.data, b.data))._parent_of((a, b))._result_of_op('@')

    def backward():
        a._update_grad(f.grad @ b.data.T)
        b._update_grad(a.data.T @ f.grad)

    return f._set_backward(backward)


@op_wrap
def tcossin(a, cos):
    f = Tensor(np.cos(a.data) if cos else np.sin(a.data))\
        ._parent_of((a,))\
        ._result_of_op('cos')

    def backward():
        a._update_grad(
            -np.sin(a.data) * f.grad if cos else np.cos(a.data) * f.grad
        )

    return f._set_backward(backward)


@op_wrap
def trelu(a):
    relu = (a.data > np.zeros(a.shape)) * a.data

    f = Tensor(relu)._parent_of((a,))._result_of_op('relu')

    def backward():
        a._update_grad(f.grad)

    return f._set_backward(backward)


@op_wrap
def tmaxmin(a, max):
    maxmin_func = np.max if max else np.min
    mask = ma.masked_equal(a.data, maxmin_func(a.data))
    
    f = Tensor(mask.fill_value)._parent_of((a, ))\
                               ._result_of_op("max" if max else "min")

    def backward():
        a._update_grad(
            mask.mask / ma.count_masked(mask) * f.grad
        )

    return f._set_backward(backward)


@op_wrap
def tsum(a):
    f = Tensor(np.sum(a.data))._parent_of((a, ))._result_of_op('sum')

    def backward():
        a._update_grad(np.ones(a.shape) * f.grad)

    f._backward = backward

    return f