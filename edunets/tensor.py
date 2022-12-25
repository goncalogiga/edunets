import numpy as np

# **** Tensor base operations ****


def op_wrap(unchanged_args=[]):
    """
    Decorator to convert non Tensor elements of the operation to Tensors
    """
    unchanged_args = [unchanged_args] if not isinstance(unchanged_args, list) else unchanged_args

    def decorator(op):
        def wrapper(*args, **kwargs):
            """ wrapper function """

            # converting values to tensors
            args = [
                arg if isinstance(arg, Tensor) or i in unchanged_args else Tensor(arg)\
                for i, arg in enumerate(args)  
            ]

            return op(*args, **kwargs)
        return wrapper
    return decorator


@op_wrap()
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

    # backward function (derivative)
    def backward():
        # Adding the gradients to avoid overwritting them when a or b is used in
        # a new operation
        a.grad += f.grad
        b.grad += f.grad

    f._backward = backward

    return f


@op_wrap()
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

    def backward():
        a.grad += b.data * f.grad
        b.grad += a.data * f.grad

    f._backward = backward

    return f


@op_wrap()
def texp(a):
    """
    The exponentiation operation is given by f(a) = e^a

    Backpropagation:
    df/da = e^a
    giving,
    dL/da = df/da * dL/df = e^a * dL/df
    """
    f = Tensor(np.exp(a))._parent_of((a, ))._result_of_op('e')

    def backward():
        a.grad += f.data * f.grad

    f._backward = backward

    return f


@op_wrap(unchanged_args=1)
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
    b_data = b.data if isinstance(b, Tensor) else b

    f = Tensor(a.data ** b_data)._result_of_op('^')
    
    f._parent_of((a, b)) if isinstance(b, Tensor) else f._parent_of((a, ))

    def backward():
        a.grad += b_data * (a.data**(b_data - 1)) * f.grad
        if isinstance(b, Tensor):
            b.grad += a.data * (b.data**(a.data - 1)) * f.grad

    f._backward = backward

    return f


# **** Tensor class definition ****


class Tensor:
    _children, _op, grad = (), "", 0


    def __init__(self, data):
        # Initiating the derivative of the tensor to nothing
        self._backward = lambda: None
        
        self.data = data


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


    # === operations based on the previous ones ===
    def __radd__(self, other): return tadd(self, other)
    def __rmul__(self, other): return tmul(self, other)

    def __sub__(self, other):
        other.data = -other.data
        return tadd(self, other)

    def __truediv__(self, other):
        return tmul(self, tpow(other, -1)._result_of_op('inv'))