import pprint
import random
import warnings
import traceback
import numpy as np
from edunets import functions as f
from edunets.visualisations import draw_dot


class Tensor:
    _op, _children, _is_leaf = "", (), False
    grad = None

    def __init__(self, data, dtype=np.float32, requires_grad=False, label=None):
        self.dtype = dtype
        self.label = label
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
        self.grad += np.nan_to_num(value) 

    
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

    
    def retain_grad(self): self.requires_grad = True

    
    def backward(self, debug=False):
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
        self.grad, prev_v = np.array(1.0), None

        # For each node, apply the chain rule to get its gradient
        for v in reversed(sorted_grah):
            if prev_v: prev_v._free_grad()

            try:
                v._backward()
            except:
                if not debug: raise
                traceback.print_exc()
                v.grad = "#Exception"

            if debug: v.requires_grad = True
            prev_v = v


    # === comparisons ===
    def __gt__(self, other): return Tensor(self.data > other.data)
    def __lt__(self, other): return Tensor(self.data < other.data)
    def __ge__(self, other): return Tensor(self.data >= other.data)
    def __le__(self, other): return Tensor(self.data <= other.data)
    def __eq__(self, other): return Tensor(self.data == other.data)
    def __ne__(self, other): return Tensor(self.data != other.data)

    # === base operations ===
    def __matmul__(self, other): return f.matmul(self, other).out
    def __add__(self, other): return f.add(self, other).out
    def __mul__(self, other): return f.mul(self, other).out
    def __pow__(self, other): return f.pow(self, other).out
    
    def cos(self): return f.cos(self).out
    def sin(self): return f.sin(self).out
    def exp(self): return f.exp(self).out
    def log(self): return f.log(self).out

    def max(self, axis=None, keepdims=False): return f.max(self, axis=axis, keepdims=keepdims).out
    def min(self, axis=None, keepdims=False): return f.min(self, axis=axis, keepdims=keepdims).out
    def sum(self, axis=None, keepdims=False): return f.sum(self, axis=axis, keepdims=keepdims).out

    def relu(self): return f.relu(self).out

    # == selection and slicing ===
    def __getitem__(self, items): 
        return f.getitem(self, items).out

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
    def softmax(self, axis):
        e = (self - Tensor(np.max(self.data, axis=axis, keepdims=True))).exp()
        return e / e.sum(axis=axis, keepdims=True)

    def mean(self): return self.sum()/sum(x for x in self.shape)
    def logsoftmax(self, axis): return self.softmax(axis).log()


    #   ~~~ activation functions ~~~
    def sigmoid(self): return 1.0/(1.0 + (-self).exp())
    def tanh(self): return 2.0 * ((2.0 * self).sigmoid()) - 1.0

    # === tensor class methods ===
    @classmethod
    def zeros_like(self, t, **kwargs): return self(np.zeros_like(t.data), **kwargs)

    @classmethod
    def zeros(self, *shape, **kwargs): return self(np.zeros(shape), **kwargs)

    @classmethod
    def ones(self, *shape, **kwargs): return self(np.ones(shape), **kwargs)

    @classmethod
    def empty(self, *shape, **kwargs): return self(np.empty(shape), **kwargs)

    @classmethod
    def randn(self, *shape, **kwargs): 
        return self(np.random.default_rng().standard_normal(size=shape), **kwargs)

    @classmethod
    def arange(self, stop, start=0, **kwargs): 
        return self(np.arange(start=start, stop=stop), **kwargs)

    @classmethod
    def uniform(self, shape, low=0.0, high=1.0, **kwargs): 
        return self(np.random.uniform(low=low, high=high, size=shape), **kwargs)

    @classmethod
    def eye(self, dim, **kwargs): return self(np.eye(dim), **kwargs)

    # === tensor propreties ===
    @property
    def graph(self): return draw_dot(self)

    @property
    def shape(self): return self.data.shape

    @property
    def T(self): self.data = self.data.T; return self 


class Function:
    def __prepare__(self, *args):
        # Convert non-Tensors to leaf Tensors
        new_args = tuple(
            arg if isinstance(arg, Tensor) else Tensor(arg, requires_grad=False)._set_as_leaf()\
            for arg in args  
        )
        return new_args if len(args) > 1 else new_args[0]


    def __init__(self, *args, brodcastable=False):
        if brodcastable: self._brodcast(*args)
        #Tensor(np.nan_to_num(self.forward()))
        self.out = Tensor(np.nan_to_num(self.forward()))\
            ._parent_of(tuple(args))\
            ._result_of_op(self.op)\
            ._set_backward(self.backward)


    def _brodcast(self, *args):
        if len(args) != 2:
            raise ValueError("_brodcast method can only deal with dual operations.")
        a, b = args[0], args[1]

        if not (a.shape == b.shape or a._is_leaf or b._is_leaf):
            try:
                brodcast_shape = np.broadcast(a.data, b.data).shape
            except ValueError:
                raise ValueError(f"Shape of tensors mismatch: {a.shape} x {b.shape}.")
        
            a_shape, b_shape = a.shape, b.shape
            
            if a_shape != brodcast_shape:
                a.data = np.broadcast_to(a.data, brodcast_shape)
            if b_shape != brodcast_shape:
                b.data = np.broadcast_to(b.data, brodcast_shape)
            
            if (a.requires_grad and a_shape != a.shape) or (b.requires_grad and b_shape != b.shape):
                warnings.warn("""Edunets can only brodcast by reshaping tensors inplace,
                beware of those changes if the brodcasted tensors are used elsewhere.""")

    
    def is_tensor(self, candidate):
        return isinstance(candidate, Tensor)


    def forward(self):
        raise RuntimeError("Forward pass was not defined.")


    def backward(self):
        raise RuntimeError("Backward pass was not defined.")