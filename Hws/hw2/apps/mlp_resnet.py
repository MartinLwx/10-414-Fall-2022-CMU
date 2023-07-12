import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
from needle.data import MNISTDataset, DataLoader
import numpy as np
import time
import os

np.random.seed(0)


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(in_features=dim, out_features=hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=drop_prob),
                nn.Linear(in_features=hidden_dim, out_features=dim),
                norm(dim),
            )
        ),
        nn.ReLU(),
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(in_features=dim, out_features=hidden_dim),
        nn.ReLU(),
        *[
            ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob)
            for _ in range(num_blocks)
        ],
        nn.Linear(in_features=hidden_dim, out_features=num_classes),
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    """Returns the average error rate (changed from accuracy) (as a float) and the average loss over all samples (as a float)"""
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
    else:
        model.eval()

    records: list[np.ndarray] = []
    wrong_prediction = 0
    criterion = nn.SoftmaxLoss()
    for X, y in dataloader:
        # (batch_size, 784)
        X = X.reshape((X.shape[0], -1))
        logits = model(X)
        loss = criterion(logits, y)
        if opt is not None:
            loss.backward()
            opt.step()
        records.append(loss.cached_data)
        wrong_prediction += (logits.numpy().argmax(-1) != y.numpy()).sum()

    return wrong_prediction / len(dataloader.dataset), np.array(records).mean()

    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    """Returns a tuple of the training error rate, training loss, test error rate, test loss computed in the last epoch of training"""
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = MNISTDataset(
        image_filename="../hw0/data/train-images-idx3-ubyte.gz",
        label_filename="../hw0/data/train-labels-idx1-ubyte.gz",
    )
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataset = MNISTDataset(
        image_filename="../hw0/data/t10k-images-idx3-ubyte.gz",
        label_filename="../hw0/data/t10k-labels-idx1-ubyte.gz",
    )
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    model = MLPResNet(784, hidden_dim=hidden_dim)
    criterion = nn.SoftmaxLoss()
    optim = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_err, train_loss, test_err, test_loss = 0.0, 0.0, 0.0, 0.0
    for i in range(epochs):
        # training
        train_err, train_loss = epoch(train_dataloader, model, optim)
        # testing
        test_err, test_loss = epoch(test_dataloader, model, None)

    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
