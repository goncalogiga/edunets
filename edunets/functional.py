import numpy as np
from edunets.tensor import Tensor

# ~~~ functions that don't require grad ~~~ #

def argmax(t, **kargs): return Tensor(np.argmax(t.data, **kargs))

# ~~~ functions that keep gradients ~~~

def convNd(N, input, weight, bias=None, stride=1, padding=0):
    ...

def conv1d(*args, **kwargs):
    ...

def conv2d(*args, **kwargs):
    ...

def conv3d(*args, **kwargs):
    ...