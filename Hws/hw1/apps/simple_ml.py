import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    # Processing images
    with gzip.open(image_filename, "rb") as zip_file:
        raw_data = zip_file.read()
    magic_number, image_cnt, row, cols = struct.unpack(">iiii", raw_data[:16])
    # 28 * 28 = 784, skip the first 16 bytes
    images = [
        struct.unpack(">" + "B" * 784, raw_data[i * 784 + 16 : (i + 1) * 784 + 16])
        for i in range(image_cnt)
    ]
    X = np.array(images, dtype=np.float32) / 255  # normalization

    # Processing labels
    with gzip.open(label_filename, "rb") as zip_file:
        raw_data = zip_file.read()
    magic_number, items = struct.unpack(">ii", raw_data[:8])
    labels = struct.unpack(">" + "B" * items, raw_data[8:])
    y = np.array(labels, dtype=np.uint8)

    return X, y
    ### END YOUR CODE


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    batch_size, _ = Z.shape
    a = ndl.log(ndl.exp(Z).sum(-1))
    b = (Z * y_one_hot).sum(-1)

    return (a - b).sum() / batch_size
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples, _ = X.shape
    iterations = num_examples // batch

    _, num_classes = W2.shape
    I_y = np.eye(num_classes)[y]

    for i in range(iterations):
        batch_X, batch_Y = (
            X[i * batch : (i + 1) * batch],
            I_y[i * batch : (i + 1) * batch],
        )  # numpy arrays
        batch_X = ndl.Tensor(batch_X)  # (batch, input_dim)
        batch_Y = ndl.Tensor(batch_Y)  # (batch, 1)

        # forward
        batch_Z1 = ndl.relu(batch_X @ W1)  # (batch, hidden_dim)
        logits = batch_Z1 @ W2  # (batch, num_classes)

        loss = softmax_loss(logits, batch_Y)
        loss.backward()

        W1 -= lr * W1.grad
        W2 -= lr * W2.grad

        W1 = W1.detach()
        W2 = W2.detach()

    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
