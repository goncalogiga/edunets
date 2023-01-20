import numpy as np
from edunets.tensor import Tensor
from edunets.utils import indexes_by_stride


# ~~~ functions returning an empty grad tensor ~~~ #


# https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
def argmax(t, **kargs): return Tensor(np.argmax(t.data, **kargs))


# ~~~ functions that update the gradient of initial tensors ~~~


def conv(input: Tensor, kernel: Tensor, stride: int=1, padding=0):
    """
    Computes the cross correlation between an input and a kernel.

    Parameters
    ----------
    + stride: int (default: 1)
        Controls the stride for the cross-correlation
    + padding: str|tuple|int (default: 0)
        Controls the amount of padding applied to the input. 
        It can be either a string {'valid', 'same'} or a tuple of ints giving the amount of padding (similar to F.pad()),
        or a single integer specifying the padding applied to every dimensions of the input.
    """
    # === padding ===
    if padding == 0:
        pass # No padding -> compute correlation right away
    elif isinstance(padding, int):
        input = input.pad((padding,)*len(input.shape))
    elif isinstance(padding, tuple):
        input = input.pad(padding)
    elif isinstance(padding, str):
        if padding == "valid":
            ... # TODO
        elif padding == "same":
            ... # TODO
        else:
            raise ValueError(f"Unknown padding mode '{padding}'. Available padding modes: 'valid', 'same'.")
    else:
        raise ValueError(f"padding should be either int, tuple or str. Received type '{type(padding)}' instead.")

    # === correlation using scipy.signal ===
    out = input.correlate(kernel)

    # === stride ===
    if stride > 1:
        # Selecting values based on stride
        idxes = indexes_by_stride(out, stride)
        out = out[idxes]

        # Defining the right shape for the correlation result
        out_shape = np.array(input.shape) - np.array(kernel.shape) + 2 - stride
        out_shape[out_shape <= 0] = 1

        # Reshape back
        out = out.reshape(out_shape)
