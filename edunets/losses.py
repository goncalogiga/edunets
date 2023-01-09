import numpy as np


class CrossEntropyLoss:
    """
    Based on https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    """
    def __init__(self, weight=None, reduction="mean"):
        self.weight = weight
        self.reduction = reduction

    def _base_l(self, x, y):
        """
        (Example) For a 3 classes and batch size 2:
        X = [[0.2, 0.2, 0.6], [0.8, 0.1, 0.1]]
        Y = [2, 1]
        x_class = [0.6, 0.1]
        return -[0.6, 0.1] + [1, 1] = [0.4, 0.9]
        """
        batch_size = y.shape[0]
        x_class = x[np.arange(batch_size), y]

        if self.weight is not None:
            return self.weight[y] * (-x[y] + x.sum(axis=1))

        return -x_class + x.exp().sum(axis=1).log()

    def forward(self, input, target):
        if self.reduction == "none":
            return self._base_l(input, target)
        elif self.reduction == "mean":
            return self._base_l(input, target).mean()
        elif self.reduction == "sum":
            return self._base_l(input, target).sum()
        else:
            raise ValueError(f"'{self.reduction}' is not a valid value for reduction")