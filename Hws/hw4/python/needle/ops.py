"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


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
                in_grad.append(init.zeros_like(value))
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
        return a + numpy.float32(self.scalar)

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
        return a * numpy.float32(self.scalar)

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

        return a.permute(new_shape)
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
        # call compact() before reshape
        return a.compact().reshape(self.shape)
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
        return array_api.broadcast_to(a, self.shape).compact()

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
        if not self.axes or len(self.axes) == 1:
            return a.sum(self.axes)
        else:
            # do summation in each axis in axes and finnaly reshape to the target shape
            # note that we may also use -1 as axis
            target_shape = []
            for idx, axis in enumerate(a.shape):
                if idx in self.axes:
                    continue
                if axis < 0:
                    axis += len(a.shape)
                target_shape.append(axis)
            target_shape = tuple(target_shape)
            # note that the target_shape may be empty

            for axis in self.axes:
                a = a.sum(axis, keepdims=True)

            return a.reshape(target_shape) if target_shape else a.reshape((1,))

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
            # i.e. we need to do summation along the first axes
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
        a = node.inputs[0]
        out = out_grad * Tensor(node.inputs[0].numpy() > 0, device=out_grad.device,dtype=out_grad.dtype)
        return out
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_max = Z.max(self.axes)
        if self.axes is None:
            Z_max_expand = Z_max.broadcast_to(Z.shape)
        else:
            target_shape = list(Z.shape)
            if isinstance(self.axes, tuple):
                for axis in self.axes:
                    target_shape[axis] = 1
            else:
                target_shape[self.axes] = 1
            Z_max_expand = array_api.broadcast_to(Z_max.reshape(target_shape), Z.shape)

        return array_api.log(array_api.exp(Z - Z_max_expand).sum(self.axes)) + Z_max

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


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 - tanh(node.inputs[0]) ** 2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        # note: we can not use np.stack here

        # 1. compute the target_shape after stacking
        target_shape = list(args[0].shape)
        target_shape.insert(self.axis, len(args))

        # 2. create a new ndl.Tensor to hold the data
        res = array_api.empty(target_shape, device=args[0].device, dtype=args[0].dtype)

        # 3. use __getitem__ to assign the arg
        # e.g. target_shape = (3, 4, 5); args[0].shape = (3, 5)
        #      (3, 1, 5) -> args[0]
        #      (3, 2, 5) -> args[1]
        #      ...
        # so we need to get the slices first
        # process_slice(self, sl, dim) will handle `None`
        lhs = tuple(slice(d) for d in args[0].shape[: self.axis])
        rhs = tuple(slice(d) for d in args[0].shape[self.axis :])
        for i, arg in enumerate(args):
            current = lhs + (slice(i, i + 1, 1),) + rhs
            res[current] = arg

        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # e.g. target_shape = (3, 4, 5); args[0].shape = (3, 5)
        # solution: split out_grad along self.axis
        out = split(out_grad, self.axis)
        return out
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        origin_shape = A.shape[: self.axis] + A.shape[self.axis + 1 :]  # skip self.axis

        lhs = tuple(slice(d) for d in A.shape[: self.axis])
        rhs = tuple(slice(d) for d in A.shape[self.axis + 1 :])
        res = []

        for i in range(A.shape[self.axis]):
            temp = array_api.empty(origin_shape, device=A.device, dtype=A.dtype)
            current = lhs + (slice(i, i + 1, 1),) + rhs
            # make sure that calling compact() before reshape()
            temp = (
                A[current].compact().reshape(origin_shape)
            )  # remove the axes = 1, e.g. (1, 5, 5) -> (5, 5)
            res.append(temp)

        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # consider an axis with size m. After dilation
        # , the size would be m * dilation + m = m * (dilation + 1)
        new_shape = list(a.shape)
        for axis in self.axes:
            new_shape[axis] = a.shape[axis] * (self.dilation + 1)
        res = array_api.empty(new_shape, device=a.device, dtype=a.dtype)
        res.fill(0)  # or you will get random number
        # now we need to copy the values from a to res
        # i.e. res[...] = a
        # decide each slice object in each axis
        indices = []
        for axis in range(len(new_shape)):
            if axis in self.axes:
                # that is, we need to change the slice here
                indices.append(slice(0, new_shape[axis], self.dilation + 1))
            else:
                indices.append(slice(0, new_shape[axis], 1))  # compact

        res[tuple(indices)] = a
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # recall the relationship: shape_before * (dilation + 1) = shape_after
        origin_shape = list(a.shape)
        for axis in self.axes:
            origin_shape[axis] = a.shape[axis] // (self.dilation + 1)

        indices = []
        for axis in range(len(a.shape)):
            if axis in self.axes:
                indices.append(slice(0, a.shape[axis], self.dilation + 1))
            else:
                indices.append(
                    slice(0, a.shape[axis], 1)
                )  # no dilate happen in this axis

        res = a[tuple(indices)].compact().reshape(origin_shape)

        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilation(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        if self.padding != 0:
            A = A.pad(
                (
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                    (0, 0),
                )
            )
            # A: N * (H + pad * 2) * (W + pad * 2) * C
        N, H, W, Cin = A.shape
        K, _, _, Cout = B.shape
        Ns, Hs, Ws, Cs = A.strides

        inner_dim = K * K * Cin
        H_out = (H - K) // self.stride + 1
        W_out = (W - K) // self.stride + 1

        A = A.as_strided(
            shape=(N, H_out, W_out, K, K, Cin),
            strides=(Ns, self.stride * Hs, self.stride * Ws, Hs, Ws, Cs),
        ).compact()

        A = A.reshape((-1, inner_dim))
        B = B.compact().reshape((-1, Cout))
        res = A @ B

        return res.compact().reshape((N, H_out, W_out, Cout))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, Weight = node.inputs
        N, H, W, Cin = X.shape
        K, _, _, Cout = Weight.shape
        #  X: N * H *  W  * Cin
        #  W: K * K * Cin * Cout
        #  out_grad: N * ((H + 2 * pad - K)//stride + 1) * ((W + 2 * pad - K)//stride + 1) * Cout

        if self.stride > 1:
            # dilation: before - m
            #           after  - m * (dilation + 1)
            #           so we pass self.stride - 1 here
            out_grad = dilate(out_grad, (1, 2), self.stride - 1)

        Weight_after = flip(Weight, (0, 1)).transpose((-2, -1))
        # Weight_after: K * K * Cout * Cin

        X_grad = conv(out_grad, Weight_after, padding=(K - self.padding - 1))
        # out_grad: N * ((H + 2 * pad - K)//stride + 1) * ((W + 2 * pad - K)//stride + 1) * Cout
        # , the output: N * [((H + 2 * pad - K)//stride + 1) + 2 * new_pad - K)//new_stride + 1]
        #                 * [((W + 2 * pad - K)//stride + 1) + 2 * new_pad - K)//new_stride + 1]
        #                 * Cin
        # goal: N * H * W * Cin
        # trivial: if stride = new_stride == 1 ---> new_pad = K - pad - 1
        # by the hints, we need to use dilation on out_grad
        # dilation and stride are opposite, that is, we can use dilation to cancle stride

        X_after = X.transpose((0, 3))
        # X_after: Cin * H * W * N
        out_grad_after = out_grad.transpose((0, 1)).transpose((1, 2))
        # out_grad_after: ((H + 2 * pad - K)//stride + 1) * ((W + 2 * pad - K)//stride + 1) * N * Cout
        W_grad = conv(X_after, out_grad_after, padding=self.padding)
        W_grad = W_grad.transpose((0, 1)).transpose((1, 2))
        # the output: Cin * [(H + 2 * new_pad - ( (H + 2 * pad - K)//stride + 1 ))//new_stride + 1]
        #                 * [(W + 2 * new_pad - ( (W + 2 * pad - K)//stride + 1 ))//new_stride + 1]
        #                 * Cout
        # goal: K * K * Cin * Cout
        # trivial: stride = new_stride == 1 ---> new_pad = pad

        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
