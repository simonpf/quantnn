"""
Test for the quantnn.data module.

"""
import os

import numpy as np
import pytest

from quantnn.data import SFTPStream


HAS_LOGIN_INFO = ("QUANTNN_SFTP_USER" in os.environ and
                  "QUANTNN_SFTP_PASSWORD" in os.environ)


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


@pytest.mark.skipif(not HAS_LOGIN_INFO, reason="No SFTP login info.")
def test_sftp_stream():
    """
    Assert that streaming via SFTP yields all data in the given folder
    and that kwargs are correctly passed on to dataset class.
    """
    stream = SFTPStream("129.16.35.202",
                        "/mnt/array1/share/MLDatasets/test/",
                        Dataset,
                        kwargs={"batch_size": 2},
                        n_workers=2)
    x_sum = 0.0
    y_sum = 0.0
    for x, y in stream:
        x_sum += x.sum()
        y_sum += y.sum()
        assert x.shape[0] == 2

    assert np.isclose(x_sum, 7 * 8 / 2 * 10 * 10)
    assert np.isclose(y_sum, 7 * 8 / 2 * 10)
