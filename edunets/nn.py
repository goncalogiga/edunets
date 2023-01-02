from edunets.tensor import Tensor

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.weight = Tensor.uniform(out_features, in_features)
        self.bias = Tensor.zeros(out_features) if bias else None

    def __call__(self, x):
        x = x * self.weight if len(self.weight.shape) == 1 else x @ self.weight.T
        return x + self.bias if self.bias is not None else x