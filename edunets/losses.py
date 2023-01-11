import typing
import numpy as np
from edunets.tensor import Tensor, TensorContent

class CrossEntropyLoss:
    """
    Based on https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    """
    def __init__(self, weight: typing.Union[TensorContent, Tensor]=None, reduction: str="mean"):
        self.weight = weight
        self.reduction = reduction

    def _base_l(self, x: Tensor, y: Tensor) -> Tensor:
        batch_size = y.shape[0]
        x_class = x[np.arange(batch_size), y]

        if self.weight is not None:
            return self.weight[y] * (-x[y] + x.exp().sum(axis=1).log())

        return -x_class + x.exp().sum(axis=1).log()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.reduction == "none":
            return self._base_l(input, target)
        elif self.reduction == "mean":
            return self._base_l(input, target).mean()
        elif self.reduction == "sum":
            return self._base_l(input, target).sum()
        else:
            raise ValueError(f"'{self.reduction}' is not a valid value for reduction")