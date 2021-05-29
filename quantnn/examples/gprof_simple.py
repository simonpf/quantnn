"""
=======================
quantnn.gprof_simple.py
=======================

This module provides download functions and dataset classes for the simple
GPROF retrieval example.
"""
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import netCDF4
import torch

from quantnn.normalizer import Normalizer


class GPROFDataset:
    """
    This class provides an interface for the training and test data for the GPROF GMI rain
    rate retrieval. It implements the torch.data.Dataset interface.
    """

    def __init__(self, path, batch_size=None, shuffle=True, normalizer=None):
        """
        Arguments:
            path(``str``): Path of the NetCDF file containing the data to load.
            batch_size: The batch size to use.
            shuffle: Whether or not to shuffle the data.
        """
        self.x = None
        self.y = None
        self.shuffle = shuffle
        self.batch_size = batch_size
        self._load_data(path)

        if normalizer:
            self.normalizer = normalizer
        else:
            self.normalizer = Normalizer(self.x)
        self.x = self.normalizer(self.x).astype(np.float32)

        self.transform_log()
        self.y = np.exp(self.y)

    def _load_data(self, path):
        """
        Load data from file into x and y attributes of the class
        instance.

        Args:
            path(``str`` or ``pathlib.Path``): Path to netCDF file containing
                the data to load.
        """

    def __len__(self):
        """
        The number of entries in the training data. This is part of the
        pytorch interface for datasets.

        Return:
            int: The number of samples in the data set
        """
        if self.batch_size is None:
            return self.x.shape[0]
        return self.x.shape[0] // self.batch_size

    def __getitem__(self, i):
        """
        Return element from the dataset. This is part of the
        pytorch interface for datasets.

        Args:
            i(int): The index of the sample to return
        """
        if self.shuffle and i == 0:
            indices = np.random.permutation(self.x.shape[0])
            self.x = self.x[indices, :]
            self.y = self.y[indices]

        if self.batch_size is None:
            return (torch.tensor(self.x[[i], :]), torch.tensor(self.y[[i]]))

        i_start = self.batch_size * i
        i_end = self.batch_size * (i + 1)
        if i >= len(self):
            raise IndexError()
        return (
            torch.tensor(self.x[i_start:i_end, :]),
            torch.tensor(self.y[i_start:i_end]),
        )

    def _load_data(self, path):
        """
        Loads data from NetCDf file.

        Arguments:
            path: Path to NetCDF file containing the data.
        """
        self.file = netCDF4.Dataset(path, mode="r")
        file = self.file

        step = 1
        tcwv = file["tcwv"][::step].data
        surface_type_data = file["surface_type"][::step].data
        t2m = file["t2m"][::step].data

        v_bt = file["brightness_temperature"]
        m, n = v_bt.shape
        bt = np.zeros((m, n), dtype=np.float32)
        index_start = 0
        chunk_size = 1024
        while index_start < m:
            index_end = index_start + chunk_size
            bt[index_start:index_end, :] = v_bt[index_start:index_end, :].data
            index_start += chunk_size
        self.y = file["surface_precipitation"][:].data
        self.y = self.y.reshape(-1, 1)

        valid = surface_type_data > 0
        valid *= t2m > 0
        valid *= tcwv > 0

        inds = np.arange(np.sum(valid))

        tcwv = tcwv[inds].reshape(-1, 1)
        t2m = t2m[inds].reshape(-1, 1)

        surface_type_data = surface_type_data[inds].reshape(-1, 1)
        surface_type_min = 1
        surface_type_max = 15
        n_classes = int(surface_type_max - surface_type_min)
        surface_type_1h = np.zeros(
            (surface_type_data.size, n_classes), dtype=np.float32
        )
        indices = (surface_type_data - surface_type_min).astype(int)
        surface_type_1h[np.arange(surface_type_1h.shape[0]), indices.ravel()] = 1.0

        bt = bt[inds]

        self.n_obs = bt.shape[-1]
        self.n_surface_classes = n_classes
        self.x = np.concatenate([bt, t2m, tcwv, surface_type_1h], axis=1)
        self.y = self.y[inds]
        self.input_features = self.x.shape[1]

    def transform_log(self):
        """
        Transforms output rain rates to log space. Samples with
        zero rain are replaced by uniformly sampling values from the
        range [0, rr.min()].
        """
        y = np.copy(self.y)
        inds = y < 1e-4
        y[inds] = 10 ** np.random.uniform(-6, -4, inds.sum())
        y = np.log(y)
        self.y = y


class GPROFTestset(GPROFDataset):
    """
    Pytorch dataset interface for the Gprof training data for the GMI sensor.

    This class is a wrapper around the netCDF4 files that are used to store
    the GProf training data. It provides as input vector the brightness
    temperatures and as output vector the surface precipitation.

    """

    def __init__(self, path, batch_size=None, normalizer=None, shuffle=True):
        """
        Create instance of the dataset from a given file path.

        Args:
            path: Path to the NetCDF4 containing the data.
            batch_size: If positive, data is provided in batches of this size
            surface_type: If positive, only samples of the given surface type
                are provided. Otherwise the surface type index is provided as
                input features.
            normalization_data: Filename of normalization data to use to
                normalize inputs.
            log_rain_rates: Boolean indicating whether or not to transform
                output to log space.
            rain_threshold: If provided, the given value is applied to the
                output rain rates to turn them into binary non-raining /
                raining labels.
        """
        super().__init__(
            path, batch_size=batch_size, normalizer=normalizer, shuffle=shuffle
        )

    def _load_data(self, path):

        self.file = netCDF4.Dataset(path, mode="r")
        file = self.file
        gprof = file["gprof"]

        step = 1
        tcwv = gprof["tcwv"][::step].data.astype(np.float32)
        surface_type_data = gprof["surface_type"][::step].data
        t2m = gprof["t2m"][::step].data.astype(np.float32)

        v_bt = gprof["brightness_temperature"]
        m, n = v_bt.shape
        bt = np.zeros((m, n))
        index_start = 0
        chunk_size = 1024
        while index_start < m:
            index_end = index_start + chunk_size
            bt[index_start:index_end, :] = v_bt[index_start:index_end, :].data
            index_start += chunk_size
        self.y_true = file["surface_precipitation"][:].data.astype(np.float32)
        self.y_true = self.y_true.reshape(-1, 1)
        self.y_gprof = gprof["surface_precipitation"][:].data.astype(np.float32)
        self.y_gprof = self.y_gprof.reshape(-1, 1)
        self.y_1st_tercile = gprof["1st_tertial"][:].data.astype(np.float32)
        self.y_3rd_tercile = gprof["2nd_tertial"][:].data.astype(np.float32)
        self.y_pop = gprof["probability_of_precipitation"][:].data.astype(np.float32)

        valid = surface_type_data > 0
        valid *= t2m > 0
        valid *= tcwv > 0
        valid *= self.y_gprof.ravel() > 0

        inds = np.where(valid)[0]
        self.inds = inds

        tcwv = tcwv[inds].reshape(-1, 1)
        t2m = t2m[inds].reshape(-1, 1)

        surface_type_data = surface_type_data[inds].reshape(-1, 1)
        surface_type_min = 1
        surface_type_max = 15
        n_classes = int(surface_type_max - surface_type_min)
        surface_type_1h = np.zeros(
            (surface_type_data.size, n_classes), dtype=np.float32
        )
        indices = (surface_type_data - surface_type_min).astype(int)
        surface_type_1h[np.arange(surface_type_1h.shape[0]), indices.ravel()] = 1.0

        bt = bt[inds]

        self.n_obs = bt.shape[-1]
        self.n_surface_classes = n_classes
        self.x = np.concatenate([bt, t2m, tcwv, surface_type_1h], axis=1)
        self.surface_type = surface_type_data
        self.y = self.y_true[inds]
        self.y_true = self.y_true[inds]
        self.input_features = self.x.shape[1]
        self.y_gprof = self.y_gprof[inds]
        self.y_pop = self.y_pop[inds]
        self.y_1st_tercile = self.y_1st_tercile[inds]
        self.y_3rd_tercile = self.y_3rd_tercile[inds]


def download_data(destination="data"):
    """
    Downloads training and evaluation data for the CTP retrieval.

    Args:
        destination: Where to store the downloaded data.
    """
    datasets = [
        "training_data_gmi_small.nc",
        "test_data_gmi_small.nc",
        "validation_data_gmi_small.nc",
    ]

    Path(destination).mkdir(exist_ok=True)
    for file in datasets:
        file_path = Path("data") / file
        if not file_path.exists():
            print(f"Downloading file {file}.")
            url = f"http://spfrnd.de/data/regn/{file}"
            urlretrieve(url, file_path)
