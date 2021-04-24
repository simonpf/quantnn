r"""
============
quantnn.data
============

This module provides generic classes to simplify the handling of training
data.
"""
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import multiprocessing
from queue import Queue
import tempfile

import numpy as np
from quantnn.common import DatasetError
from quantnn.files import CachedDataFolder, sftp

_LOGGER = logging.getLogger("quantnn.data")


def iterate_dataset(dataset):
    """
    Turns an iterable or sequence dataset into a generator.

    Returns:

        Generator object providing access to the batches in the dataset.

    Raises:

        quantnn.DatasetError when the dataset is neither iterable nor
        a sequence.

    """
    _LOGGER.info("Iterating dataset: %s", dataset)
    if isinstance(dataset, Iterable):
        yield from dataset
    elif hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        for i in range(len(dataset)):
            yield dataset[i]
    else:
        raise DatasetError("The provided dataset is neither iterable nor "
                           "a sequence.")


def open_dataset(folder,
                 path,
                 dataset_factory,
                 args=None,
                 kwargs=None):
    """
    Open a dataset.

    Args:
        folder: DatasetFolder object providing cached access to data
             files.
        path: The path of the dataset to open.
        dataset_class: The class used to read in the file.
        args: List of positional arguments passed to the dataset_factory method
              after the downloaded file.
        kwargs: Dictionary of keyword arguments passed to the dataset
            factory call.

    Returns:
        An object created using the provided dataset_factory
        using the provided args and kwargs as positional and
        keyword arguments.
    """
    if args is None:
        args = []
    if not isinstance(args, Iterable):
        raise ValueError("Provided postitional arguments 'args' must be "
                         "iterable.")
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, Mapping):
        raise ValueError("Provided postitional arguments 'kwargs' must be "
                         "a mapping.")

    _LOGGER.info("Opening dataset: %s", path)

    file = folder.get(path)
    dataset = dataset_factory(file, *args, **kwargs)
    return dataset


class DataFolder:
    """
    Interface to load and iterate over multiple dataset files in a
    folder.
    """
    def __init__(self,
                 path,
                 dataset_factory,
                 args=None,
                 kwargs=None,
                 n_workers=4,
                 n_files=None):
        """
        Create new DataFolder object.

        Args:
            path: The path of the folder containing the dataset files.
            dataset_factory: The function used to construct the dataset
                 instances for each file.
            args: Additional, positional arguments passed to
                 ``dataset_factory`` following the local file path of the
                 local copy of the dataset file.
            kwargs: Dictionary of keyword arguments passed to the dataset
                 factory.
            n_workers: The number of workers to use for concurrent loading
                 of the dataset files.
            n_files: How many of the file from the folder.
        """
        self.path = path
        self.folder = CachedDataFolder(path, n_files=n_files)
        self.dataset_factory = dataset_factory
        self.args = args
        self.kwargs = kwargs

        self.n_workers = n_workers
        self.files = self.folder.files

        # Sort datasets into random order.
        self.epoch_queue = Queue()
        self.active_queue = Queue()
        self.cache = OrderedDict()
        self.pool = ThreadPoolExecutor(max_workers=self.n_workers)
        self.folder.download(self.pool)
        self._prefetch()

    def _prefetch(self):
        if self.epoch_queue.empty():
            for f in np.random.permutation(self.files):
                self.epoch_queue.put(f)

        for i in range(self.n_workers):

            if not self.epoch_queue.empty():

                file = self.epoch_queue.get()
                self.active_queue.put(file)

                if file in self.cache:
                    continue
                else:
                    if len(self.cache) > self.n_workers:
                        self.cache.popitem(last=False)
                    arguments = [self.folder,
                                 file,
                                 self.dataset_factory,
                                 self.args,
                                 self.kwargs]
                    self.cache[file] = self.pool.submit(open_dataset,
                                                        *arguments)

    def get_next_dataset(self):
        """
        Returns the next dataset of the current epoch and issues the prefetch
        of the following data.

        Returns:
            The dataset instance that is the next in the random sequence
            of the current epoch.
        """

        #
        # Prepare download of next file.
        #

        if self.epoch_queue.empty():
            for f in np.random.permutation(self.files):
                self.epoch_queue.put(f)

        file = self.epoch_queue.get()
        self.active_queue.put(file)

        if file in self.cache:
            self.cache.move_to_end(file)
        else:
            if len(self.cache) > self.n_workers:
                self.cache.popitem(last=False)
            arguments = [self.folder, file, self.dataset_factory,
                         self.args, self.kwargs]
            self.cache[file] = self.pool.submit(open_dataset,
                                                *arguments)

        #
        # Return current file.
        #

        file = self.active_queue.get()
        dataset = self.cache[file].result()

        return dataset


    def __iter__(self):
        """
        Iterate over all batches in all remote files.
        """
        for _ in self.files:
            dataset = self.get_next_dataset()
            yield from iterate_dataset(dataset)

class LazyDataFolder:
    """
    A data folder loader for lazy datasets.
    """
    def __init__(self,
                 path,
                 dataset_factory,
                 args=None,
                 kwargs=None,
                 n_workers=4,
                 n_files=None,
                 batch_queue_size=32):
        """
        Create new DataFolder object.

        Args:
            path: The path of the folder containing the dataset files.
            dataset_factory: The function used to construct the dataset
                 instances for each file.
            args: Additional, positional arguments passed to
                 ``dataset_factory`` following the local file path of the
                 local copy of the dataset file.
            kwargs: Dictionary of keyword arguments passed to the dataset
                 factory.
            n_workers: The number of workers to use for concurrent loading
                 of the dataset files.
            n_files: How many of the file from the folder.
        """
        self.path = path
        self.folder = CachedDataFolder(path)
        self.dataset_factory = dataset_factory
        self.args = args
        self.kwargs = kwargs

        self.n_workers = n_workers
        self.files = self.folder.files

        # Sort datasets into random order.
        self.batch_queue = Queue(maxsize=batch_queue_size)
        self.pool = ProcessPoolExecutor(max_workers=self.n_workers)
        self.folder.download(self.pool)
        files = [
            self.folder.get(f) for f in np.random.permutation(self.folder.files)
        ]

        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        self.datasets = [dataset_factory(f, *args, **kwargs) for f in files]

        self.n_batches = sum([len(d) for d in self.datasets])

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        counters = {id(d): 0 for d in self.datasets}
        num_active = len(self.datasets)
        while (num_active > 0):
            for d in self.datasets:
                i = counters[id(d)]
                if i >= len(d):
                    num_active -= 1
                    continue

                # Return batch if queue is full.
                if self.batch_queue.full():
                    yield self.batch_queue.get().result()

                # Put next batch on queue.
                self.batch_queue.put(
                    self.pool.submit(d.__getitem__, i)
                )
                counters[id(d)] += 1

        while not self.batch_queue.empty():
            yield self.batch_queue.get().result()

class BatchedDataset:
    """
    A generic batched dataset, that takes two numpy array and generates a sequence
    dataset providing tensors of

    """
    def __init__(self,
                 x,
                 y,
                 batch_size=None,
                 discard_last=False,
                 tensor_backend=None,
                 shuffle=True):
        self.x = x
        self.y = y
        self.n_samples = x.shape[0]
        if batch_size is None:
            self.batch_size = 128
        else:
            self.batch_size = batch_size

        self.discard_last = False,
        self.tensor_backend = tensor_backend
        self.shuffle = shuffle

    def __len__(self):
        n_batches = self.n_samples // self.batch_size
        if (not self.discard_last) and (n_samples % self.batch_size) > 0:
            n_batches += 1
        return n_batches

    def __getitem__(self, i):

        if i >= len(self):
            raise StopIteration()

        if (i == 0) and self.shuffle:
            indices = np.random.permutation(self.n_samples)
            self.x = self.x[indices]
            self.y = self.y[indices]

        i_start = self.batch_size * i
        i_end = i_start + self.batch_size

        x_batch = self.x[i_start:i_end]
        y_batch = self.y[i_start:i_end]

        if self.tensor_backend is not None:
            x_batch = self.tensor_backend.to_tensor(x_batch)
            y_batch = self.tensor_backend.to_tensor(y_batch)

        return x_batch, y_batch
