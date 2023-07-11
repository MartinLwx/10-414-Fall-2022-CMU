import gzip
import struct
import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if not flip_img:
            return img
        return np.flip(img, 1)
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NDArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        H, W, C = img.shape
        new_img = np.zeros((H + self.padding * 2, W + self.padding * 2, C))
        new_img[
            self.padding : self.padding + H, self.padding : self.padding + W, :
        ] = img

        return new_img[
            self.padding + shift_x : self.padding + shift_x + H,
            self.padding + shift_y : self.padding + shift_y + W,
            :,
        ]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.cursor = 0
        if self.shuffle:
            orders = np.arange(len(self.dataset))
            np.random.shuffle(orders)
            self.ordering = np.array_split(
                orders, range(self.batch_size, len(self.dataset), self.batch_size)
            )
        return self
        ### END YOUR SOLUTION

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.cursor == len(self.ordering):
            raise StopIteration
        batch = [Tensor(x) for x in self.dataset[self.ordering[self.cursor]]]
        self.cursor += 1

        return batch
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        # Processing images
        with gzip.open(image_filename, "rb") as zip_file:
            raw_data = zip_file.read()
        magic_number, image_cnt, row, cols = struct.unpack(">iiii", raw_data[:16])
        # 28 * 28 = 784, skip the first 16 bytes
        images = [
            struct.unpack(">" + "B" * 784, raw_data[i * 784 + 16 : (i + 1) * 784 + 16])
            for i in range(image_cnt)
        ]

        # (N, 784)
        self.X = np.array(images, dtype=np.float32) / 255  # normalization
        self.X = self.X.reshape((-1, 28, 28, 1))

        # Processing labels
        with gzip.open(label_filename, "rb") as zip_file:
            raw_data = zip_file.read()
        magic_number, items = struct.unpack(">ii", raw_data[:8])
        labels = struct.unpack(">" + "B" * items, raw_data[8:])

        self.y = np.array(labels, dtype=np.uint8)

        self.transforms = transforms

        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        return self.apply_transforms(self.X[index]), self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
