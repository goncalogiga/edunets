import numpy as np
from edunets.tensor import Tensor

# ~~~ functions that don't require grad ~~~ #

def argmax(t, **kargs): return Tensor(np.argmax(t.data, **kargs))