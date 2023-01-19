import numpy as np


def expand_by_repeating(input, target):
    """
    Helper function
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
    selected_with_stride = {axis: [] for axis in range(len(t.shape))}

    def select_with_stride(t_, selected=[]):
        if t_.shape == () or t_.shape == (1,):
            for i, s in enumerate(selected): selected_with_stride[i].append(s)
            return

        for idx in [i for i in range(t_.shape[0]) if i % stride == 0]:
            select_with_stride(t_[idx], selected + [idx])

    select_with_stride(t); return [selected_with_stride[axis] for axis in range(len(t.shape))]