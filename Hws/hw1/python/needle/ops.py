"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * self.scalar


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # grad of a ** self.scalar -> self.scalar * (a ** (self.scalar - 1))
        return out_grad * (self.scalar * power_scalar(node.inputs[0], self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # the grad of a / b
        #    for a: 1 / b
        #    for b: - a / (b ** 2)
        # return out_grad / b, out_grad * (-a) / (b**2)
        a, b = node.inputs

        return out_grad / b, out_grad * (-a) / (b**2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return divide_scalar(out_grad, self.scalar)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_shape = list(range(len(a.shape)))
        if self.axes:
            x, y = self.axes
            new_shape[y] = x
            new_shape[x] = y
        else:
            # defaults: last two axes
            new_shape[-2], new_shape[-1] = new_shape[-1], new_shape[-2]

        return array_api.transpose(a, new_shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # size: out_grad = (a, b)
        # goal: we want (b, a)
        # solution: just transpose
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # size: <after reshape>
        # goal: <before reshape>
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # size: shape after broadcast
        # goal: original shape
        # solution: for the axes which got broadcasted, we do summation along the axes
        #           first reshape to (*origin_shape, -1) and do summation in the last axis
        origin_shape = node.inputs[0].shape

        return summation(reshape(out_grad, (*origin_shape, -1)), -1)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # size: out_grad = (m,)
        # goal: (m, n)
        # solution: by broadcast_to, we can get a (m, n) Tensor. However, we can't directly broadcast_to (m,) -> (m, n)
        #           so we first reshape (m,) -> (m, 1)
        # the summation is kind of the reverse operation of broadcast_to :)

        # cornor case: when self.axes is None, we get a scalar value
        #              , and we can just do broadcast
        origin_shape = node.inputs[0].shape
        if self.axes is None:
            return broadcast_to(out_grad, origin_shape)

        target_shape = list(origin_shape)
        if isinstance(self.axes, tuple):
            for axis in self.axes:
                target_shape[axis] = 1
        else:
            target_shape[self.axes] = 1

        return broadcast_to(reshape(out_grad, target_shape), origin_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # matmul simple
        # assumption: a = (m, n), b = (n, p), then out_grad = (m, p)
        # the grad for a @ b
        #   for a, the grad is b, and we need a matrix with size (m, n) ---> out_grad @ tranpose(b)
        #   for b, the grad is a, and we need a matrix with size (n, p) ---> transpose(a) @ out_grad

        # matmul batch mode
        # assumption: a = (b, m, n), b = (b, n, p), then out_grad = (b, m, p)
        #   for a, the grad is b, and we need a matrix with size (b, m, n) ---> out_grad @ transpose(b)
        #   for b, the grad is a, and we need a matrix with size (b, n, p) ---> transpose(a) @ out_grad

        # matmul batch mode
        # assumption: a = (b, m, n), b = (n, p), then out_grad = (b, m, p)
        #   for a, the grad is b, and we need a matrix with size (b, m, n) ---> out_grad @ transpose(b)
        #   for b, the grad is a, and we need a matrix with size (n, p)    ---> transpose(a) @ out_grad has size (b, n, p), which means we need do summation

        a, b = node.inputs

        a_grad = matmul(out_grad, transpose(b))
        b_grad = matmul(transpose(a), out_grad)

        if len(a_grad.shape) > len(a.shape):
            # i.e. we need to do summation along the
            a_grad = summation(a_grad, tuple(range(len(a_grad.shape) - len(a.shape))))
        if len(b_grad.shape) > len(b.shape):
            b_grad = summation(b_grad, tuple(range(len(b_grad.shape) - len(b.shape))))

        return a_grad, b_grad

        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -1 * a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -1 * out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a * (a > 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input = node.inputs[0]
        cached_data = input.realize_cached_data()
        return out_grad * Tensor(cached_data > 0)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
