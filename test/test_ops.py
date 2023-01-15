import torch
import unittest
import numpy as np
from edunets.tensor import Tensor

np.random.seed(0)

# acceptable difference between pytorch's results and edunets'ones.
epsilon = 0.001


def assert_forward_pass(edunets_out, pytorch_out):
    edunets_out_np = edunets_out.data
    pytorch_out_np = pytorch_out.detach().numpy()
    assert (np.abs(edunets_out_np - pytorch_out_np) < epsilon).all()


def assert_backward_pass(edunets_tensors, pytorch_tensors, edunets_out, pytorch_out):
    try:
        pytorch_out.backward(torch.ones(pytorch_out.shape))
    except:
        print("[!] Pytroch backward pass failed")
        raise

    edunets_out.backward(np.ones(edunets_out.shape))

    for i, (et, pt) in enumerate(zip(edunets_tensors, pytorch_tensors)):
        if et.requires_grad is None:
            continue
        assert (np.abs(et.grad - pt.grad.detach().numpy()) < epsilon).all()


def assert_passes(op, *data):
    edunets_tensors = [
        Tensor(el, requires_grad=True) for el in data
    ]

    pytorch_tensors = [
        torch.tensor(el, requires_grad=True) for el in data
    ]

    try:
        pytorch_out = op(*pytorch_tensors)
    except:
        print("[!] Pytorch forward pass failed.")
        raise

    edunets_out = op(*edunets_tensors)

    assert_forward_pass(edunets_out, pytorch_out)
    assert_backward_pass(edunets_tensors, pytorch_tensors, edunets_out, pytorch_out)


class TestOps(unittest.TestCase):
    a = np.random.uniform(size=(2,2))
    b = np.random.uniform(size=(2,2))
    a_flat = np.random.uniform(5,5)
    b_flat = np.random.uniform(5,5)
    c = 2.0

    # ~~~ basic ops ~~~ #

    def test_add1(self):
        assert_passes(lambda a: a + self.c, self.a)
        assert_passes(lambda a: self.c + a, self.a)

    def test_add2(self):
        assert_passes(lambda a,b: a + b, self.a, self.b)

    def test_minus1(self):
        assert_passes(lambda a: a - self.c, self.a)
        assert_passes(lambda a: self.c - a, self.a)

    def test_minus2(self):
        assert_passes(lambda a,b: a - b, self.a, self.b)

    def test_mul1(self):
        assert_passes(lambda a: a * self.c, self.a)
        assert_passes(lambda a: self.c * a, self.a)

    def test_mul2(self):
        assert_passes(lambda a,b: a * b, self.a, self.b)

    def test_div1(self):
        assert_passes(lambda a: a / self.c, self.a)

    def test_div2(self):
        assert_passes(lambda a,b: a / b, self.a, self.b)

    def test_pow1(self):
        assert_passes(lambda a: a**self.c, self.a)

    def test_pow2(self):
        assert_passes(lambda a,b: a**b, self.a, self.b)

    def test_pow3(self):
        assert_passes(lambda a,b: a**b, self.a_flat, self.b_flat)

    def test_pow4(self):
        weird_shape_a = np.random.uniform(size=(4, 3, 2, 1))
        assert_passes(lambda a: a**2, weird_shape_a)

    def test_pow5(self):
        weird_shape_a = np.random.uniform(size=(2, 2, 1))
        weird_shape_b = np.random.uniform(size=(2, 1))
        assert_passes(lambda a,b: a**b, weird_shape_a, weird_shape_b)

    def test_matmul1(self):
        assert_passes(lambda a,b: a @ b, self.a, self.b)
    
    def test_matmul2(self):
        assert_passes(lambda a,b: a @ b @ a.T @ b.T, self.a, self.b)

    def test_matmul3(self):
        m1 = np.random.uniform(size=(4,3))
        m2 = np.random.uniform(size=(3,5))
        m3 = np.random.uniform(size=(5,1))
        assert_passes(lambda m1, m2, m3: m1 @ m2 @ m3, m1, m2, m3)

    def test_getitem1(self):
        assert_passes(lambda a: a[0], self.a)

    def test_getitem2(self):
        assert_passes(lambda a: a[-1], self.a)

    def test_getitem3(self):
        assert_passes(lambda a: a[0:3], self.a)

    def test_getitem4(self):
        assert_passes(lambda a: a[a[0] > 0.0], self.a)

    def test_getitem5(self):
        assert_passes(lambda a: a[a > 0.0], self.a)

    def test_getitem6(self):
        assert_passes(lambda a: a[:, 0], self.a)
    
    def test_getitem7(self):
        assert_passes(lambda a: a[np.array(1)], self.a)

    def test_getitem8(self):
        y = np.array([1.0, 0.0])
        x = np.random.uniform(size=(4, 2))
        assert_passes(lambda x: x[np.arange(2), y], x)

    def test_getitem9(self):
        a = np.random.uniform(size=(4,4))
        b = np.random.uniform(size=(4,1))
        assert_passes(lambda a,b: (a @ b)[0:4], a, b)

    def test_getitem10(self):
        a = np.random.uniform(size=(4,4))
        b = np.random.uniform(size=(4,1))
        assert_passes(lambda a,b: b * (a @ b)[0:4], a, b)

    def test_log1(self):
        assert_passes(lambda a: a.log(), self.a)

    def test_exp1(self):
        assert_passes(lambda a: a.exp(), self.a)

    def test_cos1(self):
        assert_passes(lambda a: a.cos(), self.a)

    def test_sin1(self):
        assert_passes(lambda a: a.sin(), self.a)

    def test_tan1(self):
        assert_passes(lambda a: a.tan(), self.a)

    def test_sum1(self):
        assert_passes(lambda a: a.sum(), self.a)

    def test_sum2(self):
        assert_passes(lambda a: a.sum(dim=0), self.a)

    def test_sum3(self):
        assert_passes(lambda a: a.sum(dim=1), self.a)

    def test_sum4(self):
        assert_passes(lambda a: a.sum(dim=1, keepdims=True), self.a)

    def test_max1(self):
        assert_passes(lambda a: a.max(), self.a)

    def test_min1(self):
        assert_passes(lambda a: a.min(), self.a)

    def test_mean1(self):
        assert_passes(lambda a: a.mean(), self.a)

    # ~~~ complex ops ~~~ #

    def test_complex_ops1(self):
        a = np.random.uniform(size=(4,4))
        b = np.random.uniform(size=(4,1))

        # This is broken because brodcasting is not really
        # supported in edunets
        def op(a,b):
            return a * ((a @ b)[0:4])/((a @ b).T[-1])

        assert_passes(op, a, b)
        

if __name__ == '__main__':
    unittest.main()