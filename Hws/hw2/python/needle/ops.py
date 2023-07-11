"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


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
        return (out_grad * self.scalar,)


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
        # solution: we need to find out which axis are broadcasted, and do summation along these axis
        input_shape = node.inputs[0].shape
        output_shape = out_grad.shape

        # note: we need to figure out the shape before broadcasting, and we also need to make sure that
        # len(input_shape) = len(output_shape). Then we will find which axis is equal to 1, which means
        # we need to do summation along this axis
        # e.g. [1, 5] --- broadcast_to --- [5, 5, 5]
        #      1. create [1, 1, 5]
        #      2. find which axis is equal to 1. (0, 1)
        #      3. summation(output, (0, 1))
        if len(input_shape) != len(output_shape):
            shape_with_the_same_length = [1] * (
                len(output_shape) - len(input_shape)
            ) + list(input_shape)
        else:
            shape_with_the_same_length = input_shape

        axes = []
        for idx, val in enumerate(shape_with_the_same_length):
            if val == 1:
                axes.append(idx)

        return summation(out_grad, tuple(axes))
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
        input_shape = node.inputs[0].shape
        if self.axes is None:
            return broadcast_to(out_grad, input_shape)

        target_shape = list(input_shape)
        if isinstance(self.axes, tuple):
            for axis in self.axes:
                target_shape[axis] = 1
        else:
            target_shape[self.axes] = 1

        return broadcast_to(reshape(out_grad, target_shape), input_shape)
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


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION

        # exp(Z - Z_max) requires they have the same shape
        Z_max = array_api.max(Z, self.axes)
        if self.axes is None:
            Z_max_expand = array_api.broadcast_to(Z_max, Z.shape)
        else:
            target_shape = list(Z.shape)
            if isinstance(self.axes, tuple):
                for axis in self.axes:
                    target_shape[axis] = 1
            else:
                target_shape[self.axes] = 1
            Z_max_expand = array_api.broadcast_to(Z_max.reshape(target_shape), Z.shape)

        return (
            array_api.log(array_api.sum(array_api.exp(Z - Z_max_expand), self.axes))
            + Z_max
        )

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # size: out_grad = <after-max>
        # goal: <before-max>

        # 1. compute the partial derivative - softmax(Z - Z_max)
        # see: https://en.wikipedia.org/wiki/LogSumExp, we just substitute x with Z-Z_MAX
        Z = node.inputs[0]
        Z_max = Tensor(array_api.max(Z.cached_data, self.axes, keepdims=True))
        up = exp(Z - Z_max)
        down = summation(up, self.axes)

        # 2. reshape out_grad and down s.t. they have the same length with Z.
        #    then use broadcast_to s.t out_grad.shape = Z.shape and down.shape = Z.shape
        input_shape = Z.shape
        if self.axes is None:
            shape_with_the_same_length = input_shape
            out_grad = broadcast_to(out_grad, shape_with_the_same_length)
            down = broadcast_to(down, shape_with_the_same_length)
        else:
            shape_with_the_same_length = list(Z.shape)
            if isinstance(self.axes, tuple):
                for axis in self.axes:
                    shape_with_the_same_length[axis] = 1
            else:
                shape_with_the_same_length[self.axes] = 1
            out_grad = broadcast_to(
                out_grad.reshape(shape_with_the_same_length), input_shape
            )
            down = broadcast_to(down.reshape(shape_with_the_same_length), input_shape)

        return out_grad * (up / down)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
