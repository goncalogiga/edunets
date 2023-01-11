import pprint
import random
import typing
import warnings
import traceback
import numpy as np
from edunets import functions as f
from edunets.visualisations import draw_dot

# Types accepted by the Tensor constructor
TensorContent = typing.Union[int, float, np.ndarray, typing.List[float]]
TensorOpArgs = typing.Union[int, float, np.ndarray, typing.List[float], 'Tensor']


class Tensor:
    """
    Multi-dimensional matrix containing elements of a single data type.

    A class used to store a numpy array of a single data type and track the
    operations the array is involved in. This is so it can be differentiated 
    if method `backward` is called.

    Attributes
    ----------
    + data : np.ndarray
        numpy array representing the matrix containing elements of a single data type.
    + requires_grad : bool
        if set to true the class will track operations made on the tensor so its gradient
        can be calculated when method `backward` is called.
    + label : str
        a name given to the tensor when displayed in the operations graph returned by method `grad`

    Methods
    -------
    + keep_nans()
        prevents the use of np.nan_to_num during calculations
    + retain_grad()
        same thing as setting requires_grad to true
    + zero_grad()
        empties the gradient of the tensor (setting it to None)
    + backward(debug=False)
        performs backpropagation using automatic differentiation on the
        operation graph built from the tensor. This updates the value of self._grad.
    + grad()
        a proprety used to access the numpy array stored in self._grad
    + shape()
        a proprety used to access the shape of the numpy array stored in self.data

    plus all methods involving calculations on the data (mean, sum, cos, ect.)
    and methods similar to numpy to create usual matrices (eye, zeros, ect.)

    Private Attributes
    ------------------
    + _op : str
        a string storing the operation responsible for the creation of
        the present tensor
    + _NaNs : bool
        if this is true, np.nan_to_num will never be called
    + _children : tuple
        a tuple storing the tensors involved in the operation responsible for
        the creation of the present tensor
    + _is_leaf : bool
        if true this tensor will never have its gradient computed (usefull for constant tensors)
    + _grad: np.ndarray
        numpy array representing the gradient of the matrix stored in self.data.
    + _key: int
        a unique integer linked to the tensor used during topological sort of the
        operations graph (in self.backward()).
    + _backward: Callable([], None)
        a function that updates the gradients of the children of this tensor using
        its gradient (chain rule)
    """
    _op: str = ""
    _NaNs: bool = False
    _children: tuple = ()
    _is_leaf: bool = False
    _grad: np.ndarray = None


    def __init__(self, data: TensorContent, dtype: type=np.float32, requires_grad: bool=False, label: str=None):
        """
        Parameters
        ----------
        + data : np.ndarray
            numpy array representing the matrix containing elements of a single data type.
        + dtype:
            type of the numpy array representing the matrix
        + requires_grad : bool
            if set to true the class will track operations made on the tensor so its gradient
            can be calculated when method `backward` is called.
        + label : str
            a name given to the tensor when displayed in the operations graph returned by method `grad`
        """
        self.dtype: type = dtype
        self.label: str = label
        self._key: int = random.randint(0, 2**32)
        self.requires_grad: bool = bool(requires_grad)
        self.data: np.ndarray = np.array(data, dtype=dtype)
        self._backward: typing.Callable[[], None] = lambda: None

        if self.data.shape == (): self.data = np.expand_dims(self.data, axis=0)


    def __repr__(self) -> str:
        """
        Representation of the tensor when print is called.
        """
        if self.data.shape == (1,):
            data_repr = pprint.pformat(self.data)[7:-1].replace(']', '')
        else:
            data_repr = pprint.pformat(self.data)[6:-1]

        if self.requires_grad:
            return f"tensor({data_repr}, requires_grad=True)"
        else:
            return f"tensor({data_repr})"


    def _update_grad(self, value: np.ndarray) -> None:
        """
        Updates the gradient of the tensor

        Parameters
        ----------
        + value: np.ndarray
            new value given to the gradient of the tensor
        """
        if self._is_leaf: return self
        if self._grad is None: self._grad = 0
        self._grad += value if self._NaNs else np.nan_to_num(value)

    
    def _free_grad(self) -> None:
        """
        Sets the gradient of the tensor to None.
        """
        if self.requires_grad is False: self._grad = None


    def _update_nans_opt(self, children: tuple) -> None:
        """
        Propagates the choice of keeping nans in the matrices if one of 
        the children (tensors that resulted in the present tensor from 
        an operation) has self._NaNs set to True. 

        Parameters
        ----------
        + children: tuple
            a tuple storing the tensors involved in the operation responsible for
            the creation of the present tensor
        """
        if not any(c._NaNs for c in children):
            self.data = np.nan_to_num(self.data)
        else:
            self._NaNs = True


    def _parent_of(self, children: tuple) -> 'Tensor':
        """
        Stores the children (tensors that resulted in the present tensor from 
        an operation) of this tensor

        Parameters
        ----------
        + children: tuple
            a tuple storing the tensors involved in the operation responsible for
            the creation of the present tensor
        """
        self._update_nans_opt(children)
        self._children = children
        return self

    
    def _result_of_op(self, op: str) -> 'Tensor':
        """
        Stores the operation used to create the present tensor so it can be
        displayed in the operation graph returned by the proprety `self.graph`.

        Parameters
        ----------
        + op : str
            a string storing the operation responsible for the creation of
            the present tensor (example: '+', '*', 'log')
        """
        self._op = op
        return self


    def _set_as_leaf(self) -> 'Tensor':
        """
        Sets this tensor as a 'leaf' tensor (no gradients will be computed)
        """
        self._is_leaf = True
        return self


    def _set_backward(self, backward: typing.Callable[[], None]) -> 'Tensor':
        """
        Sets this tensor's backward function, a function which will be called
        during the backpropagation of the operation graph.
        
        Parameters
        ----------
        + _backward: Callable([], None)
            a function that updates the gradients of the children of this tensor using
            its gradient (chain rule)
        """
        self._backward = backward
        return self

    # All these methods are explained in the Tensor class' docstring
    def keep_nans(self) -> None: self._NaNs = True
    def zero_grad(self) -> None: self._grad = None
    def retain_grad(self) -> None: self.requires_grad = True
    
    def backward(self, debug: bool=False) -> None:
        """
        Backpropagation algorithm

        This function topologicaly sorts the operation graph so chain rules
        can be computed in the right order. The chain rule is calculated insise
        of every nodes `_backward` attributes. This backward function calculates
        the gradient ('derivative') of tensors engaged in operations (defined in
        the functions.py file) and is the core to the automatic differentiation
        engine this class is all about.   

        Parameters
        ----------
        + debug: bool
            if set to true backpropagation will proceed even if a backward() call
            raised an Exception. The traceback of the exception is still printed
            and the graph of operations shows where the gradients could not
            be computed (obtained via self.graph).

        Raises
        ------
        RuntimeError
            If the tensor contains a matrix instead of a scalar
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
        self._grad, prev_v = np.array(1.0), None

        # For each node, apply the chain rule to get its gradient
        for v in reversed(sorted_grah):
            if prev_v: prev_v._free_grad()

            try:
                v._backward()
            except:
                if not debug: raise
                traceback.print_exc()
                v._grad = "#Exception"

            if debug: v.requires_grad = True
            prev_v = v


    # === comparisons ===
    def __gt__(self, other: 'Tensor') -> 'Tensor': return Tensor(self.data > other.data)
    def __lt__(self, other: 'Tensor') -> 'Tensor': return Tensor(self.data < other.data)
    def __ge__(self, other: 'Tensor') -> 'Tensor': return Tensor(self.data >= other.data)
    def __le__(self, other: 'Tensor') -> 'Tensor': return Tensor(self.data <= other.data)
    def __eq__(self, other: 'Tensor') -> 'Tensor': return Tensor(self.data == other.data)
    def __ne__(self, other: 'Tensor') -> 'Tensor': return Tensor(self.data != other.data)

    # === base operations ===
    def __add__(self, other: TensorOpArgs) -> 'Tensor': return f.add(self, other).out
    def __mul__(self, other: TensorOpArgs) -> 'Tensor': return f.mul(self, other).out
    def __pow__(self, other: TensorOpArgs) -> 'Tensor': return f.pow(self, other).out
    def __matmul__(self, other: TensorOpArgs) -> 'Tensor': return f.matmul(self, other).out
    
    def cos(self) -> 'Tensor': return f.cos(self).out
    def sin(self) -> 'Tensor': return f.sin(self).out
    def exp(self) -> 'Tensor': return f.exp(self).out
    def log(self) -> 'Tensor': return f.log(self).out

    def max(self, axis: typing.Tuple[int, ...]=None, keepdims: bool=False) -> 'Tensor': return f.max(self, axis=axis, keepdims=keepdims).out
    def min(self, axis: typing.Tuple[int, ...]=None, keepdims: bool=False) -> 'Tensor': return f.min(self, axis=axis, keepdims=keepdims).out
    def sum(self, axis: typing.Tuple[int, ...]=None, keepdims: bool=False) -> 'Tensor': return f.sum(self, axis=axis, keepdims=keepdims).out

    def relu(self) -> 'Tensor': return f.relu(self).out

    # == selection and slicing ===
    def __getitem__(self, items: typing.Union[int, slice, typing.List[int], np.ndarray, 'Tensor']) -> 'Tensor': 
        return f.getitem(self, items).out

    # === operations based on the previous ones ===
    def __sub__(self, other: TensorOpArgs) -> 'Tensor': return self + (-other)
    def __truediv__(self, other: TensorOpArgs) -> 'Tensor': return self * other**(-1)

    def __rpow__(self, other: TensorContent) -> 'Tensor': return other**self
    def __radd__(self, other: TensorContent) -> 'Tensor': return self + other
    def __rmul__(self, other: TensorContent) -> 'Tensor': return self * other
    def __rsub__(self, other: TensorContent) -> 'Tensor': return -self + other
    def __rtruediv__(self, other: TensorContent) -> 'Tensor': return other * self**(-1)

    def __neg__(self) -> 'Tensor': return -1 * self
    def tan(self) -> 'Tensor': return self.sin()/self.cos()

    # === more advanced operations ===
    def softmax(self, axis: typing.Tuple[int, ...]) -> 'Tensor':
        e = (self - Tensor(np.max(self.data, axis=axis, keepdims=True))).exp()
        return e / e.sum(axis=axis, keepdims=True)

    def mean(self) -> 'Tensor': return self.sum()/sum(x for x in self.shape)
    def logsoftmax(self, axis: typing.Tuple[int, ...]) -> 'Tensor': return self.softmax(axis).log()

    #   ~~~ activation functions ~~~
    def sigmoid(self) -> 'Tensor': return 1.0/(1.0 + (-self).exp())
    def tanh(self) -> 'Tensor': return 2.0 * ((2.0 * self).sigmoid()) - 1.0

    # === tensor class methods ===
    @classmethod
    def zeros_like(self, t: 'Tensor', **kwargs) -> 'Tensor': return self(np.zeros_like(t.data), **kwargs)

    @classmethod
    def zeros(self, *shape, **kwargs) -> 'Tensor': return self(np.zeros(shape), **kwargs)

    @classmethod
    def ones(self, *shape, **kwargs) -> 'Tensor': return self(np.ones(shape), **kwargs)

    @classmethod
    def empty(self, *shape, **kwargs) -> 'Tensor': return self(np.empty(shape), **kwargs)

    @classmethod
    def randn(self, *shape, **kwargs) -> 'Tensor': 
        return self(np.random.default_rng().standard_normal(size=shape), **kwargs)

    @classmethod
    def arange(self, stop: int, start: int=0, **kwargs) -> 'Tensor': 
        return self(np.arange(start=start, stop=stop), **kwargs)

    @classmethod
    def uniform(self, *shape, low: float=0.0, high: float=1.0, **kwargs) -> 'Tensor': 
        return self(np.random.uniform(low=low, high=high, size=shape), **kwargs)

    @classmethod
    def eye(self, dim: int, **kwargs) -> 'Tensor': return self(np.eye(dim), **kwargs)

    # === tensor propreties ===
    @property
    def graph(self) -> 'Digraph': return draw_dot(self)

    @property
    def shape(self) -> typing.Tuple[int, ...]: return self.data.shape

    @property
    def T(self) -> 'Tensor': self.data = self.data.T; return self

    @property
    def grad(self) -> np.ndarray:
        if self._grad is None: 
            warnings.warn(f"Access of empty gradient: tensor(shape={self.shape}, requires_grad={self.requires_grad}, label={self.label})")
        return self._grad


class Function:
    """
    A class that every tensor operation class should inherit from

    Attributes
    ----------
    + brodcastable : bool
        ...
    + out: Tensor
        ...

    Methods
    -------
    + forward()
        ...
    + backward()
        ...
    + is_tensor()
        ...
    """
    def __prepare__(self, *args) -> typing.Union[Tensor, typing.Tuple[Tensor]]:
        """
        
        """
        new_args = tuple(
            arg if isinstance(arg, Tensor) else Tensor(arg, requires_grad=False)._set_as_leaf()\
            for arg in args  
        )
        return new_args if len(args) > 1 else new_args[0]


    def __init__(self, *args, brodcastable: bool=False) -> 'Function':
        """
        
        """
        if brodcastable: self._brodcast(*args)

        self.out = Tensor(self.forward())\
            ._parent_of(tuple(args))\
            ._result_of_op(self.op)\
            ._set_backward(self.backward)


    def _brodcast(self, *args) -> None:
        """
        
        """
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

    
    def is_tensor(self, candidate: typing.Any) -> bool:
        """
        
        """
        return isinstance(candidate, Tensor)


    def forward(self) -> None:
        """
        
        """
        raise RuntimeError("Forward pass was not defined.")


    def backward(self) -> None:
        """
        
        """
        raise RuntimeError("Backward pass was not defined.")


# Tensor constructor to avoid circular imports
def to_tensor(data: TensorContent, **kargs) -> Tensor: return Tensor(data, **kargs)