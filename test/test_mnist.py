import unittest
import warnings
import numpy as np
from edunets.functional import argmax
from edunets.tensor import Tensor
from edunets.losses import CrossEntropyLoss


np.random.seed(0)

# Expected accuracy
expected_acc = 0.8


class EduNet1:
    def __init__(self):
        # https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073
        stdv1, stdv2 = 1./np.sqrt(28*28), 1./np.sqrt(128)
        self.l1 = Tensor.uniform(128, 28*28, low=-stdv1, high=stdv1, requires_grad=True, label="L1")
        self.l2 = Tensor.uniform(10, 128, low=-stdv2, high=stdv2, requires_grad=True, label="L1")

    def __call__(self, x):
        x = x @ self.l1.T
        x = x.relu()
        x = x @ self.l2.T
        return x


class SGD:
    def __init__(self, params, lr=0.001):
        self.lr = lr
        self.params = params

    def step(self):
        for t in self.params:
            t.data = t.data - t.grad * self.lr

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()


def fetch(url):
    import requests, gzip, os, hashlib, numpy
    fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        with open(fp, "wb") as f:
            dat = requests.get(url).content
            f.write(dat)
    return numpy.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()


class TestMNIST(unittest.TestCase):
    def test_mnist_from_linear_layers(self):
        X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
        Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]

        model = EduNet1()
        batch_size = 256
        loss_fn = CrossEntropyLoss(reduction="mean")
        optim = SGD([model.l1, model.l2], lr=0.001)
        epochs = 100

        for _ in range(epochs):
            samp = np.random.randint(0, X_train.shape[0], size=(batch_size))

            X = Tensor(X_train[samp].reshape((-1, 28*28)))
            Y = Tensor(Y_train[samp])

            out = model(X)

            cat = argmax(out, axis=1)
            
            accuracy = (cat == Y).mean()

            loss = loss_fn.forward(out, Y)
            
            optim.zero_grad()
            
            loss.backward()
            
            optim.step()
            
            if accuracy.data > expected_acc:
                return

        raise AssertionError(f"Accuracy never met expected accuracy of {expected_acc}.")


if __name__ == '__main__':
    with warnings.catch_warnings():
        unittest.main()