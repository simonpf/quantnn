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
import queue
import tempfile

import numpy as np
from quantnn.common import DatasetError
from quantnn.files import CachedDataFolder, sftp

_LOGGER = logging.getLogger("quantnn.data")


class ActiveDataset(multiprocessing.Process):
    """
    The active dataset class takes care of concurrent reading of
    data from a dataset.
    """
    def __init__(self,
                 factory,
                 filename,
                 queue,
                 args=None,
                 kwargs=None):
        """
        Args:
            factory: Class or factory function to use to open the dataset.
            filename: Filename of the dataset file to open.
            batch_queue: Queue on which to put the loaded batches.
            args: List of positional arguments to pass to the dataset factory
                following the dataset name.
            kwargs: Dictionary of keyword arguments to pass to the dataset factory
                following the dataset name.
        """
        super().__init__()
        self.factory = factory
        self.filename = filename
        self.queue = queue
        if args is None:
            self.args = []
        else:
            self.args = args
        if kwargs is None:
            self.kwargs = []
        else:
            self.kwargs = kwargs

    def run(self):
        """
        Open dataset and start loading batches.
        """
        dataset = self.factory(self.filename, *self.args, **self.kwargs)
        if isinstance(dataset, Iterable):
            for b in dataset:
                self.queue.put(b)
        elif hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
            for i in range(len(dataset)):
                self.queue.put(dataset[i])
        del dataset
        self.queue.join()


class DataFolder:
    """
    Utility class that iterates over a folder containing multiple files with training
    data. Data is loaded concurrently from a given number of processes and batches
    are returned in round robin manner from currently active processes.

    Attributes:
        path: The path of the folder containing the datasets to load.
        folder: 'CachedDataFolder' instance providing access to the files in
             the folder.
        dataset_factory: The class or factory function to instantiate dataset objects
             for the files in the folder.
        args: List of additional positional arguments passed to 'dataset_factory'
        kwargs: List of additional keyword arguments passed to 'dataset_factory'
        files: List of (local) filenames of the datafiles in the folder.
        active_datasets: List of the processes that load the data from currently active
            datasets.
        queue_size: The number of batches in each active datasets queue.
        n_files: If provided, will be used to limit the loaded files to the first
            'n_files' found in the folder.
    """
    def __init__(
        self,
        path,
        dataset_factory,
        args=None,
        kwargs=None,
        active_datasets=4,
        queue_size=16,
        n_files=None
    ):
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
        self.queue_size = queue_size

        pool = ThreadPoolExecutor(max_workers=active_datasets)
        self.folder.download(pool)

        self.files = [self.folder.get(f) for f in self.folder.files]
        self.active_datasets = min(active_datasets, len(self.files))
        self.current_epoch = list(np.random.permutation(self.files))
        self.next_epoch = list(np.random.permutation(self.files))
        self.queues = [multiprocessing.JoinableQueue(queue_size)
                       for i in range(active_datasets)]
        self.datasets = []
        for i, _ in enumerate(self.files[:active_datasets]):
            p = ActiveDataset(self.dataset_factory,
                              self.current_epoch.pop(),
                              self.queues[i],
                              args=self.args,
                              kwargs=self.kwargs)
            p.start()
            self.datasets.append(p)
        self.next_datasets = []
        self.next_queues = []

    def __del__(self):
        for d in self.datasets:
            d.terminate()


    def __iter__(self):
        """
        Iterate over all batches in all remote files.
        """
        while True:
            all_exhausted = True
            all_empty = True
            for i in range(len(self.datasets)):
                p = self.datasets[i]
                q = self.queues[i]

                # Check if a new dataset should be loaded.
                if not p.is_alive():
                    if self.current_epoch:
                        p = ActiveDataset(self.dataset_factory,
                                          self.current_epoch.pop(),
                                          self.queues[i],
                                          args=self.args,
                                          kwargs=self.kwargs)
                        p.start()
                        self.datasets[i] = p
                    else:
                        if len(self.next_datasets) < self.active_datasets:
                            self.next_queues.append(
                                multiprocessing.JoinableQueue(self.queue_size)
                            )
                            p = ActiveDataset(self.dataset_factory,
                                              self.next_epoch.pop(),
                                              self.next_queues[-1],
                                              args=self.args,
                                              kwargs=self.kwargs)
                            p.start()
                            self.next_datasets.append(p)
                else:
                    all_exhausted = False

                # Get batch from queue.
                try:
                    b = q.get_nowait()
                except queue.Empty:
                    continue
                q.task_done()
                all_empty = False
                yield b

            if all_exhausted and all_empty:
                self.queues = self.next_queues
                self.datasets = self.next_datasets
                self.current_epoch = self.next_epoch
                self.next_epoch = list(np.random.permutation(self.files))
                self.next_queues = []
                self.next_datasets = []
                break


class LazyDataFolder:
    """
    A data folder loader for lazy datasets.
    """

    def __init__(
        self,
        path,
        dataset_factory,
        args=None,
        kwargs=None,
        n_workers=4,
        n_files=None,
        batch_queue_size=32,
    ):
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
        files = [self.folder.get(f) for f in np.random.permutation(self.folder.files)]

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
        while num_active > 0:
            for d in self.datasets:
                i = counters[id(d)]
                if i >= len(d):
                    num_active -= 1
                    continue

                # Return batch if queue is full.
                if self.batch_queue.full():
                    yield self.batch_queue.get().result()

                # Put next batch on queue.
                self.batch_queue.put(self.pool.submit(d.__getitem__, i))
                counters[id(d)] += 1

        while not self.batch_queue.empty():
            yield self.batch_queue.get().result()


class BatchedDataset:
    """
    A generic batched dataset, that takes two numpy array and generates a sequence
    dataset providing tensors of

    """

    def __init__(
        self,
        x,
        y,
        batch_size=None,
        discard_last=False,
        tensor_backend=None,
        shuffle=True,
    ):
        self.x = x
        self.y = y
        self.n_samples = x.shape[0]
        if batch_size is None:
            self.batch_size = 128
        else:
            self.batch_size = batch_size

        self.discard_last = (False,)
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
