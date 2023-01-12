import warnings
import numpy as np


class Function:
    """
    A class that every tensor operation class should inherit from

    A Function is some form of operation (unary, binary or other) that can
    be applied to Tensors. The Function class wrapps every new operation so
    that a Function simply has to define a forward and a backward method.
    Forward computes the actual operation while backward updates the gradients
    of the Tensors with respect to the type of operation that is being defined.

    Attributes
    ----------
    + brodcastable : bool
        if set to true, tensors will be expanded in order to match the shape of
        the other Tensor it is being computed with.
        Example: [2] * [1, 1, 1] will be changed to [2, 2, 2] * [1, 1, 1] prior
        to the actual multiplication of matrices
    + out: Tensor
        out contains the actual numeric result of the forward pass in the form of a Tensor.

    Methods
    -------
    + forward()
        forward needs to be implemented by every function taking tensors as arguments
    + backward()
        backard needs to be implemented so the backpropagation algorithm can work 
    """
    def __init__(self, *args, brodcastable: bool=False):
        """
        Function initialization that creates the result of the operation
        and stores it into the `self.out` attribute.
        
        Parameters
        ----------
        + args:
            A tuple of all the arguments taken by the function
        + brodcastable:
            if set to true, tensors will be expanded in order to match the shape of
            the other Tensor it is being computed with.
            Example: [2] * [1, 1, 1] will be changed to [2, 2, 2] * [1, 1, 1] prior
            to the actual multiplication of matrices
        """
        from edunets.tensor import Tensor
        self.Tensor = Tensor

        if brodcastable: self._brodcast(*args)

        self.out = Tensor(self.forward())\
            ._parent_of(tuple(args))\
            ._result_of_op(self.op)\
            ._set_backward(self.backward)


    def __prepare__(self, *args):
        """
        Converts all arguments of the function to Tensors
        Example: [2] + Tensor([2]) becomes Tensor([2]) + Tensor([2]) thanks to
        this function. In addition, The constants transformed into Tensors
        are set to static Tensors so no gradients are computed (self._is_static=True).
        
        Parameters
        ----------
        + args:
            A tuple of all the arguments taken by the function
        """
        from edunets.tensor import Tensor

        new_args = tuple(
            arg if isinstance(arg, Tensor) else Tensor(arg, requires_grad=False)._set_as_static()\
            for arg in args  
        )
        return new_args if len(args) > 1 else new_args[0]


    def _brodcast(self, *args) -> None:
        """
        Tensors will be expanded in order to match the shape of 
        the other Tensor it is being computed with.
        
        Example: [2] * [1, 1, 1] is changed to [2, 2, 2] * [1, 1, 1] thanks
        to this function.

        [!] This is only done for Tensors that require a gradient.
        
        Parameters
        ----------
        + args:
            A tuple of all the arguments taken by the function

        Raises
        ------
        ValueError
            If brodcasting is done for anything other then binary operations
            OR
            If brodcasting is impossible because of a shape missmatch
        """
        if len(args) != 2:
            raise ValueError("_brodcast method can only deal with dual operations.")
        a, b = args[0], args[1]

        if not (a.shape == b.shape or a._is_static or b._is_static):
            try:
                brodcast_shape = np.broadcast(a.data, b.data).shape
            except ValueError:
                raise ValueError(f"Shape of tensors mismatch: {a.shape} x {b.shape}.")
        
            a_shape, b_shape = a.shape, b.shape
            
            if a_shape != brodcast_shape:
                a.data = np.broadcast_to(a.data, brodcast_shape)
            if b_shape != brodcast_shape:
                b.data = np.broadcast_to(b.data, brodcast_shape)
            
            if (a.requires_grad and a_shape != a.shape) or (b.requires_grad and b_shape != b.shape):
                warnings.warn("""Edunets can only brodcast by reshaping tensors inplace,
                beware of those changes if the brodcasted tensors are used elsewhere.""")


    # These should be overwritten during a Function definition
    def forward(self) -> None:
        raise RuntimeError("Forward pass was not defined.")
    def backward(self) -> None:
        raise RuntimeError("Backward pass was not defined.")