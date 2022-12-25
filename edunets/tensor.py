import numpy as np
from edunets.visualisations import draw_dot


class Tensor:
    _children, _op, grad = (), "", 0


    def __init__(self, data, requires_grad=True):
        # Initiating the derivative of the tensor to nothing
        self._backward = lambda: None
        
        self.data = data
        self.requires_grad = requires_grad


    def __repr__(self):
        return f"tensor({self.data})"


    def _parent_of(self, children):
        """
        Used to keep track of the graph of operations applied to the tensor
        """
        self._children = set(children)
        return self

    
    def _result_of_op(self, op):
        self._op = op
        return self

    
    def backward(self):
        """
        Backpropagation algorithm
        """
        sorted_grah, visited = [], set()
        
        def topological_sort(v):
            if v not in visited:
                visited.add(v)

                for child in v._children:
                    topological_sort(child)

                sorted_grah.append(v)

        # Topological sort of the graph of operations
        topological_sort(self)

        # Initial gradient (last node) is set to 1.0 so we can calculate the first
        # chain rule result.
        # Backpropagation is done assuming the node calling .backward()
        # is the last of the operation grah
        self.grad = 1.0

        # For each node, apply the chain rule to get its gradient
        for v in reversed(sorted_grah):
            v._backward()


    # === basic operations ===
    def __add__(self, other): return tadd(self, other)
    def __mul__(self, other): return tmul(self, other)
    def __pow__(self, other): return tpow(self, other)

    # === operations based on the previous ones ===
    def __radd__(self, other): return tadd(self, other)
    def __rmul__(self, other): return tmul(self, other)
    def __neg__(self): return tmul(self, -1)
    def __sub__(self, other): return tadd(self, self.__neg__(other))
    def __rsub__(self, other): return tadd(self.__neg__(), other)
    def __truediv__(self, other): return tmul(self, tpow(other, -1))
    def __rpow__(self, other): return tpow(other, self)

    # === more advanced operations ===
    def exp(self): return texp(self)

    # === display graph of operations ===
    def graph(self):
        return draw_dot(self)



# **** Tensor base operations ****

def op_wrap(op):
    """
    Decorator to convert constants to Tensor constants
    """
    def wrapper(*args, **kwargs):
        # converting values to tensors
        args = [
            arg if isinstance(arg, Tensor) else Tensor(arg, requires_grad=False)\
            for arg in args  
        ]
        return op(*args, **kwargs)
    return wrapper


def op_backward(*args):
    """
    Generates the backward function for the operation.
    Excpected arguments are:
    op_backward(a, _backward_a, b, _backward_b) ect...

    return a _backward function object
    """
    backfuncs = [
        tensor_backfunc\
        for (tensor, tensor_backfunc) in args if tensor.requires_grad
    ]

    def _backward():
        for tensor_backfunc in backfuncs: tensor_backfunc() 
    
    return _backward


@op_wrap
def tadd(a, b):
    """
    The addition operation is given by f(a,b) = a+b

    a ---
        + --- res --- op --- L
    b ---

    Backpropagation:
    df/da = 1.0
    df/db = 1.0
    giving,
    dL/da = df/da * dL/df = dL/df (=gradient of f)
    dL/db = df/db * dL/df = dL/df (=gradient of f)
    """
    f = Tensor(a.data + b.data)._parent_of((a, b))._result_of_op('+')

    # backward function for a (derivative)
    def backward_a():
        # Adding the gradients to avoid overwritting them when a is used in
        # a new operation
        a.grad += f.grad
    
    # same thing for b
    def backward_b():
        b.grad += f.grad

    f._backward = op_backward((a, backward_a), (b, backward_b))

    return f


@op_wrap
def tmul(a, b):
    """
    The multiplication operation is given by f(a,b) = a*b

    Backpropagation:
    df/da = b
    df/db = a
    giving,
    dL/da = df/da * dL/df = b * dL/df
    dL/db = df/db * dL/df = a * dL/df
    """
    f = Tensor(a.data * b.data)._parent_of((a, b))._result_of_op('*')

    def backward_a():
        a.grad += b.data * f.grad
    
    def backward_b():
        b.grad += a.data * f.grad

    f._backward = op_backward((a, backward_a), (b, backward_b))

    return f


@op_wrap
def texp(a):
    """
    The exponentiation operation is given by f(a) = e^a

    Backpropagation:
    df/da = e^a
    giving,
    dL/da = df/da * dL/df = e^a * dL/df
    """
    f = Tensor(np.exp(a.data))._parent_of((a, ))._result_of_op('e')

    def backward():
        a.grad += f.data * f.grad

    f._backward = op_backward((a, backward))

    return f


@op_wrap
def tpow(a, b):
    """
    The power operation is given by f(a,b) = a^b

    Backpropagation:
    df/da = b*a**(b-1)
    df/db = a*b**(a-1)
    giving,
    dL/da = df/da * dL/df = b*a**(b-1) * dL/df
    dL/db = df/db * dL/df = a*b**(a-1) * dL/df
    """
    f = Tensor(a.data ** b.data)._parent_of((a, b))._result_of_op('**')

    def backward_a():
        a.grad += b.data * (a.data**(b.data - 1)) * f.grad
    
    def backward_b():
        b.grad += a.data * (b.data**(a.data - 1)) * f.grad

    f._backward = op_backward((a, backward_a), (b, backward_b))

    return f