import numpy as np
from edunets.tensor import Tensor


def argmax(t, **kargs): return Tensor(np.argmax(t.data, **kargs))

def one_hot(t): 
    b = np.zeros((t.data.size, int(t.data.max() + 1)), dtype=int)
    b[np.arange(t.data.size), t.data.astype(int)] = 1.0
    return Tensor(b)