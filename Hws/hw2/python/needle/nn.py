"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        configs = {
            "device": device,
            "dtype": dtype,
            "requires_grad": True,
        }
        self.has_bias = bias
        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, **configs)
        )
        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(out_features, 1, **configs).reshape(
                    (1, out_features)
                )
            )

        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.has_bias:
            return X @ self.weight + self.bias.broadcast_to(
                (X.shape[0], self.out_features)
            )
        else:
            return X @ self.weight
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return X.reshape((X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)

        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # Note: y is a list of numbers
        batch_size, num_classes = logits.shape
        labels = init.one_hot(num_classes, y)
        out = ops.logsumexp(logits, -1)

        return (out - ops.summation(logits * labels, -1)).sum() / batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        configs = {
            "device": device,
            "dtype": dtype,
        }
        self.weight = Parameter(init.ones(self.dim, **configs, requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim, **configs, requires_grad=True))
        self.running_mean = init.zeros(self.dim, **configs)
        self.running_var = init.ones(self.dim, **configs)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mean = x.sum(0) / x.shape[0]
            # print(f"shape of mean: {mean.shape}, vs {x.shape}")
            E_X = ops.broadcast_to(mean.reshape((1, -1)), x.shape)
            # print(f"shape of E_X: {E_X.shape}")
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean.data

            var = ((x - E_X) ** 2).sum(0) / x.shape[0]
            # print(f"shape of var: {var.shape}, vs {x.shape}")
            Var_X = ops.broadcast_to(var.reshape((1, -1)), x.shape)
            # print(f"shape of Var_X: {Var_X.shape}")
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var.data

            return self.weight * (x - E_X) / ((Var_X + self.eps) ** (1 / 2)) + self.bias

        else:
            return (x - self.running_mean) / ((self.running_var + self.eps) ** (1 / 2))
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps

        ### BEGIN YOUR SOLUTION
        configs = {
            "device": device,
            "dtype": dtype,
            "requires_grad": True,
        }

        self.weight = Parameter(init.ones(self.dim, **configs))
        self.bias = Parameter(init.zeros(self.dim, **configs))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # size: x = (N, d)
        E_X = x.sum(-1) / x.shape[1]
        E_X = ops.broadcast_to(E_X.reshape((x.shape[0], -1)), x.shape)

        Var_X = ops.broadcast_to(
            ((x - E_X) ** 2 / x.shape[1]).sum(-1).reshape((x.shape[0], -1)), x.shape
        )

        return self.weight * (x - E_X) / ((Var_X + self.eps) ** (1 / 2)) + self.bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # multiply each axis
        target_length = 1
        for i in x.shape:
            target_length *= i
        rand_mask = init.randb(target_length, p=self.p).reshape(x.shape)

        if not self.training:
            return x
        else:
            return x * rand_mask * (1 / (1 - self.p))
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
