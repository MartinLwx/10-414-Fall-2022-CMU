import sys

sys.path.append("./python")
import needle as ndl
import needle.nn as nn
import math
import numpy as np

np.random.seed(0)


class ConvBN(ndl.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        configs = {
            "device": device,
            "dtype": dtype,
        }
        self.conv2d = nn.Conv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            bias=True,
            **configs,
        )
        self.bn = nn.BatchNorm2d(out_channels, **configs)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv2d(x)))


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        configs = {
            "device": device,
            "dtype": dtype,
        }
        self.network = nn.Sequential(
            ConvBN(3, 16, 7, 4, **configs),
            ConvBN(16, 32, 3, 2, **configs),
            nn.Residual(
                nn.Sequential(
                    ConvBN(32, 32, 3, 1, **configs),
                    ConvBN(32, 32, 3, 1, **configs),
                )
            ),
            ConvBN(32, 64, 3, 2, **configs),
            ConvBN(64, 128, 3, 2, **configs),
            nn.Residual(
                nn.Sequential(
                    ConvBN(128, 128, 3, 1, **configs),
                    ConvBN(128, 128, 3, 1, **configs),
                )
            ),
            nn.Flatten(),
            nn.Linear(128, 128, bias=True, **configs),
            nn.ReLU(),
            nn.Linear(128, 10, bias=True, **configs),
        )
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.network(x)
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(
        self,
        embedding_size,
        output_size,
        hidden_size,
        num_layers=1,
        seq_model="rnn",
        device=None,
        dtype="float32",
    ):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        configs = {
            "device": device,
            "dtype": dtype,
        }
        self.emb = nn.Embedding(output_size, embedding_size, **configs)
        if seq_model == "rnn":
            self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers, **configs)
        elif seq_model == "lstm":
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers, **configs)

        self.linear = nn.Linear(hidden_size, output_size, **configs)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        emb = self.emb(x)  # (seq_len, bs, embedding_size)
        output, h_t = self.seq_model(emb, h)  # output: (seq_len, bs, hidden_size)
        output = self.linear(
            output.reshape((seq_len * bs, -1))
        )  # output: (seq_len, bs, hidden_size)

        return output, h_t
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset(
        "data/cifar-10-batches-py", train=True
    )
    train_loader = ndl.data.DataLoader(
        cifar10_train_dataset, 128, ndl.cpu(), dtype="float32"
    )
    print(dataset[1][0].shape)
