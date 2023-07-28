"""The module.
"""
from typing import List
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
            return x @ self.weight + self.bias.broadcast_to(
                (x.shape[0], self.out_features)
            )
        else:
            return x @ self.weight
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return X.compact().reshape((X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
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
        # Note: to avoid promotion, we'd better use broadcast_to extensively
        if self.training:
            mean = x.sum(0) / x.shape[0]
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean.data
            mean = ops.broadcast_to(mean.reshape((1, -1)), x.shape)

            var = ((x - mean) ** 2).sum(0) / x.shape[0]
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var.data
            var = ops.broadcast_to(var.reshape((1, -1)), x.shape)

        else:
            mean = ops.broadcast_to(self.running_mean.reshape((-1, self.dim)), x.shape)
            var = ops.broadcast_to(self.running_var.reshape((-1, self.dim)), x.shape)
        weight = ops.broadcast_to(self.weight.reshape((-1, self.dim)), x.shape)
        bias = ops.broadcast_to(self.bias.reshape((-1, self.dim)), x.shape)

        equation = (x - mean) / (var + self.eps) ** (1 / 2)

        return weight * equation + bias
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2, 3)).transpose((1, 2))


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
        E_X = x.sum(-1) / x.shape[1]
        E_X = ops.broadcast_to(E_X.reshape((x.shape[0], -1)), x.shape)

        Var_X = ops.broadcast_to(
            ((x - E_X) ** 2 / x.shape[1]).sum(-1).reshape((x.shape[0], -1)), x.shape
        )

        weight = ops.broadcast_to(self.weight.reshape((1, self.dim)), x.shape)
        bias = ops.broadcast_to(self.bias.reshape((1, self.dim)), x.shape)

        return weight * (x - E_X) / ((Var_X + self.eps) ** (1 / 2)) + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x
        else:
            rand_mask = init.randb(*x.shape, p=1 - self.p) / (1 - self.p)

            return x * rand_mask
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.bias = bias
        configs = {
            "device": device,
            "dtype": dtype,
            "requires_grad": True,
        }
        # weight_shape: (K, K, Cin, Cout)
        weight_shape = (kernel_size, kernel_size, in_channels, out_channels)
        self.weight = Parameter(
            init.kaiming_uniform(in_channels, out_channels, weight_shape, **configs)
        )

        if bias:
            a = 1 / ((in_channels * kernel_size**2) ** 0.5)
            self.bias = Parameter(
                init.rand(out_channels * 1, low=-a, high=a, **configs)
            )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x: N * C * H * W -> N * H * W * C
        N, C, H, W = x.shape
        x = x.transpose((1, 2)).transpose((2, 3))

        # compute the padding term
        pad = self.kernel_size // 2
        if self.bias:
            res = ops.conv(x, self.weight, stride=self.stride, padding=pad)
            bias = self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(
                res.shape
            )
            res += bias
        else:
            res = ops.conv(x, self.weight, stride=self.stride, padding=pad)

        return res.transpose((2, 3)).transpose((1, 2))
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()

        ### BEGIN YOUR SOLUTION
        self.bias = bias
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        configs = {
            "device": device,
            "dtype": dtype,
            "requires_grad": True,
        }
        a = (1 / hidden_size) ** 0.5

        self.W_ih = Parameter(
            init.rand(input_size, hidden_size, low=-a, high=a, **configs)
        )
        self.W_hh = Parameter(
            init.rand(hidden_size, hidden_size, low=-a, high=a, **configs)
        )
        if bias:
            self.bias_hh = Parameter(init.rand(hidden_size, low=-a, high=a, **configs))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-a, high=a, **configs))

        if nonlinearity == "tanh":
            self.activation_fn = Tanh()
        elif nonlinearity == "relu":
            self.activation_fn = ReLU()
        else:
            raise ValueError("Not support")
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor containing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)

        if self.bias:
            out = (
                X @ self.W_ih
                + h @ self.W_hh
                + self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(
                    (bs, self.hidden_size)
                )
                + self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(
                    (bs, self.hidden_size)
                )
            )
        else:
            out = X @ self.W_ih + h @ self.W_hh

        out = self.activation_fn(out)
        return out
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.rnn_cells = []
        for _ in range(num_layers):
            self.rnn_cells.append(
                RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)
            )
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # BPTT
        seq_len, bs, input_size = X.shape
        if h0 is None:
            h0 = init.zeros(
                self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype
            )
        h_n = []

        hidden_states = ops.split(h0, 0)
        # each one: (bs, hidden_size)

        for i in range(self.num_layers):
            temp = []
            h = hidden_states[i]
            input_states = ops.split(X, 0)
            # each_one: (bs, input_size)
            for t in range(seq_len):
                h = self.rnn_cells[i](input_states[t], h)
                # h: (bs, hidden_size)
                temp.append(h)
                # collect last hidden state in this layer
                if t == seq_len - 1:
                    h_n.append(h)
            X = ops.stack(temp, 0)
            # the input X^l of the l-th layers is the hidden states from last layer
            # X: (seq_len, bs, input_size)
        return X, ops.stack(h_n, 0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(
        self, input_size, hidden_size, bias=True, device=None, dtype="float32"
    ):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
