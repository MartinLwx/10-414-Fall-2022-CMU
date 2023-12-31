import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
from models import *
from tqdm import tqdm
import time

device = ndl.cpu()

### CIFAR-10 training ###


def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
    else:
        model.eval()
    records: list[np.ndarray] = []
    total_acc = 0.0
    iterations = len(dataloader.dataset) // dataloader.batch_size
    pbar = tqdm(dataloader, total=iterations)
    for X, y in pbar:
        X, y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
        logits = model(X)
        loss = loss_fn(logits, y)
        if opt is not None:
            loss.backward()
            opt.step()
        records.append(float(loss.numpy()))
        total_acc += (logits.numpy().argmax(-1) == y.numpy()).sum()
        pbar.set_postfix(loss=sum(records) / len(records))

    return total_acc / len(dataloader.dataset), sum(records) / len(records)
    ### END YOUR SOLUTION


def train_cifar10(
    model,
    dataloader,
    n_epochs=1,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    loss_fn=nn.SoftmaxLoss,
):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    criterion = loss_fn()
    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(n_epochs):
        train_avg_acc, train_avg_loss = epoch_general_cifar10(
            dataloader, model, criterion, optimizer
        )

    return train_avg_acc, train_avg_loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    return epoch_general_cifar10(dataloader, model, loss_fn(), opt=None)
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(
    data,
    model,
    seq_len=40,
    loss_fn=nn.SoftmaxLoss(),
    opt=None,
    clip=None,
    device=None,
    dtype="float32",
):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
    else:
        model.eval()

    nbatch, _ = data.shape
    total_acc, total_loss = 0.0, 0.0
    cnt = 0
    h = None
    pbar = tqdm(range(0, nbatch, seq_len), total=nbatch // seq_len)
    for i in pbar:
        # construct the batch data
        batch, target = ndl.data.get_batch(data, i, seq_len, device, dtype)
        # data - Tensor of shape (bptt, bs) with cached data as NDArray
        # target - Tensor of shape (bptt*bs, ) with cached data as NDArray

        output, h = model(batch, h)
        # output: (seq_len*bs, output_size)
        # h: (num_layers, bs, hidden_size) if using RNN
        # h: a tuple of (h0, c0), each of shape (num_layers, bs, hidden_size) if using LSTM

        if isinstance(h, tuple):
            h = (h[0].detach(), h[1].detach())
        else:
            h = h.detach()

        loss = loss_fn(output, target)

        total_loss += loss.numpy() * batch.shape[1]

        if opt is not None:
            loss.backward()
            opt.step()

        total_acc += (target.numpy() == output.numpy().argmax(-1)).sum()
        cnt += batch.shape[1]
        pbar.set_postfix(loss=total_loss.sum() / cnt)

    return total_acc / cnt, total_loss.sum() / cnt

    ### END YOUR SOLUTION


def train_ptb(
    model,
    data,
    seq_len=40,
    n_epochs=1,
    optimizer=ndl.optim.SGD,
    lr=4.0,
    weight_decay=0.0,
    loss_fn=nn.SoftmaxLoss,
    clip=None,
    device=None,
    dtype="float32",
):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)

    ### BEGIN YOUR SOLUTION
    criterion = loss_fn()
    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    avg_acc, avg_loss = 0.0, 0.0
    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(
            data,
            model,
            seq_len,
            criterion,
            opt=optimizer,
            clip=clip,
            device=device,
            dtype=dtype,
        )

    return avg_acc, avg_loss

    ### END YOUR SOLUTION


def evaluate_ptb(
    model, data, seq_len=40, loss_fn=nn.SoftmaxLoss, device=None, dtype="float32"
):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    return epoch_general_ptb(
        data, model, seq_len, loss_fn(), opt=None, device=device, dtype=dtype
    )
    ### END YOUR SOLUTION


if __name__ == "__main__":
    ### For testing purposes
    device = ndl.cpu()
    # dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    # dataloader = ndl.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True
    #         )
    #
    # model = ResNet9(device=device, dtype="float32")
    # train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
    #      lr=0.001, weight_decay=0.001)

    corpus = ndl.data.Corpus("./data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = ndl.data.batchify(
        corpus.train, batch_size, device=device, dtype="float32"
    )
    model = LanguageModel(
        1, len(corpus.dictionary), hidden_size, num_layers=2, device=device
    )
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
