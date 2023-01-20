import typing
import numpy as np
from edunets.tensor import Tensor
from edunets.utils import indexes_by_stride


# ~~~ functions returning an empty grad tensor ~~~ #


# https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
def argmax(t, **kargs): return Tensor(np.argmax(t.data, **kargs))


# ~~~ functions that update the gradient of initial tensors ~~~


def conv(input: Tensor, kernel: Tensor, stride: int=1, padding_mode: str="constant",
         padding: typing.Union[int, tuple, str]=0, dilation: typing.Union[int, tuple]=1):
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
    + padding_mode: str (default: "constant")
        One of the following modes defined in the numpy pad function:
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    + dilation: int|tuple (default: 1)
        Spacing between kernel elements. 
    """
    chosen_padding = lambda x, pad_width: x.pad(pad_width, mode=padding_mode)

    # === padding ===
     # padding='valid' is the same as no padding
    if padding != 0 and padding != "valid":
        # ~~ Squeezing input before padding ~~
        # We remove the non-empty dimensions and replace them with -1,
        # this is so we can update the shapes after the padding
        input_shape_model = [-1 if s > 1 else 1 for s in input.shape]
        input = input.squeeze()

        if isinstance(padding, int):
            input = chosen_padding(input, (padding,)*input.dim)
        elif isinstance(padding, tuple):
            input = chosen_padding(input, padding)
        elif isinstance(padding, str):
            if padding == "same":
                ... # TODO
            else:
                raise ValueError(f"Unknown padding mode '{padding}'. Available padding modes: 'valid', 'same'.")
        else:
            raise ValueError(f"padding should be either int, tuple or str. Received type '{type(padding)}' instead.")

        # Adding the removed dimensions
        k = 0
        for i, s in enumerate(input_shape_model):
            if s == -1:
                input_shape_model[i] = input.shape[k]
                k += 1

        input = input.reshape(tuple(input_shape_model))

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

    return out