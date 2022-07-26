"""
Test for the quantnn.data module.

"""
import logging
import os

import numpy as np
import pytest
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from quantnn.data import DataFolder, LazyDataFolder


# Currently no SFTP test data available.
HAS_LOGIN_INFO = False


LOGGER = logging.getLogger(__file__)


class Dataset:
    """
    A test dataset class to test the streaming of data via SFTP.
    """
    def __init__(self,
                 filename,
                 batch_size=1):
        """
        Create new dataset.

        Args:
           filename: Path of the file load the data from.
           batch_size: The batch size of the samples to return.
        """

        self.batch_size = batch_size
        data = np.load(filename)
        self.x = data["x"]
        self.y = data["y"].reshape(-1, 1)
        LOGGER.info("Loaded data from file %s.", filename)

    def _shuffle(self):
        """
        Shuffles the data order keeping x and y samples consistent.
        """
        indices = np.random.permutation(self.x.shape[0])
        self.x = self.x[indices]
        self.y = self.y[indices]

    def __len__(self):
        """ Number of samples in dataset. """
        return self.x.shape[0] // self.batch_size

    def __getitem__(self, index):
        """ Return batch from dataset. """
        if index >= len(self):
            raise IndexError()

        if index == 0:
            self._shuffle()

        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        return (self.x[start:end], self.y[start:end])


@pytest.mark.xfail()
@pytest.mark.skipif(not HAS_LOGIN_INFO, reason="No SFTP login info.")
def test_sftp_stream():
    """
    Assert that streaming via SFTP yields all data in the given folder
    and that kwargs are correctly passed on to dataset class.
    """
    host = "129.16.35.202"
    path = "/mnt/array1/share/MLDatasets/test/"
    stream = DataFolder("sftp://" + host + path,
                        Dataset,
                        kwargs={"batch_size": 2},
                        aggregate=None,
                        n_workers=16)
    stream_2 = DataFolder("sftp://" + host + path,
                          Dataset,
                          kwargs={"batch_size": 2},
                          n_workers=16)

    next(iter(stream))
    next(iter(stream_2))

    x_sum = 0.0
    y_sum = 0.0
    for x, y in stream:
        x_sum += x.sum()
        y_sum += y.sum()
        assert x.shape[0] == 2

    x_sum = 0.0
    y_sum = 0.0
    for x, y in stream_2:
        x_sum += x.sum()
        y_sum += y.sum()
        assert x.shape[0] == 2

    x_sum = 0.0
    y_sum = 0.0
    for x, y in stream:
        x_sum += x.sum()
        y_sum += y.sum()
        assert x.shape[0] == 2

    x_sum = 0.0
    y_sum = 0.0
    for i, (x, y) in enumerate(stream_2):
        x_sum += x.sum()
        y_sum += y.sum()
        assert x.shape[0] == 2

    assert np.isclose(x_sum, 7 * 8 / 2 * 10 * 10)
    assert np.isclose(y_sum, 7 * 8 / 2 * 10)


@pytest.mark.skipif(not HAS_LOGIN_INFO, reason="No SFTP login info.")
def test_lazy_datafolder():
    host = "129.16.35.202"
    path = "/mnt/array1/share/MLDatasets/test/"
    stream = LazyDataFolder("sftp://" + host + path,
                            Dataset,
                            kwargs={"batch_size": 2},
                            n_workers=2,
                            batch_queue_size=1)

    x_sum = 0.0
    y_sum = 0.0
    for x, y in stream:
        x_sum += x.sum()
        y_sum += y.sum()
        assert x.shape[0] == 2

    assert np.isclose(x_sum, 7 * 8 / 2 * 10 * 10)
    assert np.isclose(y_sum, 7 * 8 / 2 * 10)

class TensorDataset:
    """
    A test dataset class to test the streaming of data via SFTP.
    """
    def __init__(self,
                 filename,
                 batch_size=1):
        """
        Create new dataset.

        Args:
           filename: Path of the file load the data from.
           batch_size: The batch size of the samples to return.
        """

        self.batch_size = batch_size
        data = np.load(filename)
        self.x = data["x"]
        self.y = data["y"].reshape(-1, 1)

    def _shuffle(self):
        """
        Shuffles the data order keeping x and y samples consistent.
        """
        indices = np.random.permutation(self.x.shape[0])
        self.x = self.x[indices]
        self.y = self.y[indices]

    def __len__(self):
        """ Number of samples in dataset. """
        return self.x.shape[0] // self.batch_size

    def __getitem__(self, index):
        """ Return batch from dataset. """
        if index >= len(self):
            raise IndexError()

        if index == 0:
            self._shuffle()

        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        return (torch.tensor(self.x[start:end]),
                torch.tensor(self.y[start:end]))


@pytest.mark.skipif(not HAS_LOGIN_INFO, reason="No SFTP login info.")
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed.")
def test_aggregation():
    """
    Assert that aggregation of tensor works as expected.
    """
    host = "129.16.35.202"
    path = "/mnt/array1/share/MLDatasets/test/"
    stream = DataFolder("sftp://" + host + path,
                        TensorDataset,
                        kwargs={"batch_size": 1},
                        aggregate=2,
                        n_workers=16)
    stream_2 = DataFolder("sftp://" + host + path,
                          TensorDataset,
                          kwargs={"batch_size": 1},
                          aggregate=2,
                          n_workers=16)

    next(iter(stream))
    next(iter(stream_2))

    x_sum = 0.0
    y_sum = 0.0
    for x, y in stream:
        x_sum += x.sum()
        y_sum += y.sum()
        assert x.shape[0] == 2

    x_sum = 0.0
    y_sum = 0.0
    for x, y in stream_2:
        x_sum += x.sum()
        y_sum += y.sum()
        assert np.all(np.isclose(x[:, 0].detach().numpy().ravel(),
                                 y.detach().numpy().ravel()
        ))
        assert x.shape[0] == 2

    x_sum = 0.0
    y_sum = 0.0
    for x, y in stream:
        x_sum += x.sum()
        y_sum += y.sum()
        assert x.shape[0] == 2

    x_sum = 0.0
    y_sum = 0.0

    for x, y in stream_2:
        x_sum += x.sum()
        y_sum += y.sum()
        assert x.shape[0] == 2

    assert np.isclose(x_sum, 7 * 8 / 2 * 10 * 10)
    assert np.isclose(y_sum, 7 * 8 / 2 * 10)
