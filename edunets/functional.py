import numpy as np
from edunets.tensor import Tensor


def argmax(t, **kargs): return Tensor(np.argmax(t.data, **kargs))