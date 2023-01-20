import numpy as np


def expand_by_repeating(input, target):
    """
    Helper function used in backpropagation
    """
    
    input_dim, target_dim = len(input.shape), len(target.shape)
    dim_diff = target_dim - input_dim

    input = np.expand_dims(input, axis=[target_dim - i - 1 for i in range(dim_diff)])

    expansion = [
        (dt, axis)\
        for axis, (di, dt) in enumerate(zip(input.shape, target.shape)) if di != dt
    ]

    result = input
    for repeats, axis in expansion:
        result = np.repeat(result, repeats=repeats, axis=axis)

    return result


def indexes_by_stride(t, stride=1):
    """
    Helper function used in the 'conv' function (when using stride > 1)
    """

    selected_with_stride = {axis: [] for axis in range(len(t.shape))}

    def select_with_stride(t_, selected=[]):
        if t_.shape == () or t_.shape == (1,):
            for i, s in enumerate(selected): selected_with_stride[i].append(s)
            return

        for idx in [i for i in range(t_.shape[0]) if i % stride == 0]:
            select_with_stride(t_[idx], selected + [idx])

    select_with_stride(t); return tuple(selected_with_stride[axis] for axis in range(len(t.shape)))


def unpad(data: np.ndarray, pad_width: tuple):
    """
    Helper function used in backpropagation
    Inspired by https://porespy.org/_modules/porespy/tools/_unpad.html#unpad
    """

    pad_width = np.asarray(pad_width).squeeze()

    if pad_width.ndim == 0:
        pad_width = np.array(pad_width for _ in range(0, len(data.shape)))

    if pad_width.ndim > 2:
        raise ValueError("pad_width is too large to unpad. Maximum dimension available is 2.")

    if pad_width.ndim == 1:
        shape = data.shape - pad_width[0] - pad_width[1]
        if shape[0] < 1:
            shape = np.array(data.shape) * shape
        s_im = []
        for dim in range(data.ndim):
            lower_im = pad_width[0]
            upper_im = shape[dim] + pad_width[0]
            s_im.append(slice(int(lower_im), int(upper_im)))

    if pad_width.ndim == 2:
        shape = np.asarray(data.shape)
        s_im = []
        for dim in range(data.ndim):
            shape[dim] = data.shape[dim] - pad_width[dim][0] - pad_width[dim][1]
            lower_im = pad_width[dim][0]
            upper_im = shape[dim] + pad_width[dim][0]
            s_im.append(slice(int(lower_im), int(upper_im)))

    return data[tuple(s_im)]