from collections import OrderedDict
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
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
        print(path)
        self.path = path
        self.folder = CachedDataFolder(path)
        self.dataset_factory = dataset_factory
        self.args = args
        self.kwargs = kwargs

        self.n_workers = n_workers
        self.files = self.folder.files

        # Sort datasets into random order.
        self.epoch_queue = Queue()
        self.active_queue = Queue()
        self.cache = OrderedDict()
        self.pool = ProcessPoolExecutor(max_workers=self.n_workers)
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
