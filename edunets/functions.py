import numpy as np
from edunets.tensor import Tensor


def argmax(t, **kargs):
    t.data = np.argmax(t.data, **kargs)
    return t