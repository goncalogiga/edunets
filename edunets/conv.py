import numpy as np
from edunets.tensor import Tensor
from numpy.lib.stride_tricks import sliding_window_view


class ConvBuilder:
    def __init__(self, input, kernel, padding="", stride=""):
        self.input = input
        self.kernel = kernel
        self.stride = stride
        self.kernel_size = self.kernel.shape


    def slice_input_to_kernel_size(self):
        return (
            Tensor(arr)\
            for arr in sliding_window_view(self.input, self.kernel_size)
        )


    def map_to_slice(self, slice, map):
        """
        Applies the function 'map' to every tensor in slice (list of tensors)
        """
        pass


    def slice_to_array(self, slice):
        pass