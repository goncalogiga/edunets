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