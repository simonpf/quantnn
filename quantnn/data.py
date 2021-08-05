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
import os
from queue import Queue
import queue
import tempfile
from time import sleep

import numpy as np
from quantnn.common import DatasetError
from quantnn.files import CachedDataFolder, sftp
from quantnn.backends import get_tensor_backend
from quantnn import utils

_LOGGER = logging.getLogger("quantnn.data")

def split(data, n):
    return (data[i:i + n] for i in range(0, len(data), n))


class DatasetLoader(multiprocessing.Process):
    """
    The active dataset class takes care of concurrent reading of
    data from a dataset.
    """
    def __init__(self,
                 factory,
                 queue_size,
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
        self.file_queue = multiprocessing.Queue()
        self.batch_queue = multiprocessing.JoinableQueue(queue_size)
        if args is None:
            self.args = []
        else:
            self.args = args
        if kwargs is None:
            self.kwargs = []
        else:
            self.kwargs = kwargs

        # Initialize done flag to set indicating
        # no work to be done.
        self.done_flag = multiprocessing.Event()
        self.done_flag.set()

    def run(self):
        """
        Open dataset and start loading batches.
        """
        while True:

            # Wait while done flag is set.
            while self.done_flag.is_set():
                sleep(0.01)

            # Process files

            files = self.file_queue.get()

            for filename in files:
                dataset = self.factory(filename, *self.args, **self.kwargs)
                if isinstance(dataset, Iterable):
                    for b in dataset:
                        self.batch_queue.put(b)
                elif (hasattr(dataset, "__len__") and
                        hasattr(dataset, "__getitem__")):
                    for i in range(len(dataset)):
                        self.batch_queue.put(dataset[i])
                del dataset

            # Processing finished.
            self.batch_queue.join()
            self.done_flag.set()


class DatasetManager(multiprocessing.Process):
    """
    Manager process that fetches and potentially aggregates batches
    produces by loader processes.
    """
    def __init__(self,
                 dataset_factory,
                 files,
                 n_workers,
                 queue_size,
                 aggregate=None,
                 shuffle=True,
                 args=None,
                 kwargs=None):
        super().__init__()
        self.dataset_factory = dataset_factory
        self.files = files
        self.n_workers = n_workers
        self.queue_size = n_workers * queue_size
        self.aggregate = aggregate
        self.shuffle = shuffle
        self.args = args
        self.kwargs = kwargs
        self.batch_queue = multiprocessing.JoinableQueue(queue_size)
        self.backend = None

        # Event for communication with master process.
        self.restart_flag = multiprocessing.Event()
        self.restart_flag.clear()
        self.done_flag = multiprocessing.Event()
        self.done_flag.clear()

        seed = int.from_bytes(os.urandom(4), "big") + os.getpid()
        self._rng = np.random.default_rng(seed)

        # Create initialize and start workers.
        files = list(self._rng.permutation(self.files))
        n = len(files) // self.n_workers
        if len(files) % self.n_workers > 0:
            n += 1
        self.workers = []
        for fs in split(files, n):
            worker = DatasetLoader(self.dataset_factory,
                                   self.queue_size,
                                   args=self.args,
                                   kwargs=self.kwargs)
            worker.daemon = True
            worker.file_queue.put(fs)
            worker.done_flag.clear()
            worker.start()
            self.workers.append(worker)

    def aggregate_batches(self, batches):
        """
        Aggregate list of batches.

        Args:
            batches: List of batches to aggregate.

        Return:
            Tuple ``(x, y)`` containing the aggregated inputs and outputs in
            'batches'.
        """
        xs = []
        ys = None
        # Collect batches.
        for x, y in batches:
            xs.append(x)
            if isinstance(y, dict):
                if ys is None:
                    ys = {}
                for k, y in y.items():
                    ys.setdefault(k, []).append(y)
            else:
                if ys is None:
                    ys = []
                ys.append(y)

        if self.backend is None:
            self.backend = get_tensor_backend(xs[0])

        x = self.backend.concatenate(xs, 0)
        y = utils.apply(lambda y: self.backend.concatenate(y, 0), ys)

        if self.shuffle:
            indices = self._rng.permutation(x.shape[0])
            f = lambda x: x[indices]
            x = f(x)
            y = utils.apply(f, y)
        return x, y

    def run(self):
        """
        Collects batches from child process and puts them on the batch
        queue.
        """
        while True:

            batches = []
            # Collect batches from workers.
            while not all([w.done_flag.is_set() for w in self.workers]):
                for w in self.workers:
                    # Get batch from queue.
                    try:
                        b = w.batch_queue.get_nowait()
                        w.batch_queue.task_done()
                        batches.append(b)
                    except FileNotFoundError as e:
                        _LOGGER.warning(
                            "FileNotFoundError occured when retrieving batch from "
                            "loader process.", e
                        )
                        continue
                    except queue.Empty:
                        continue

                    if self.aggregate is None:
                        for b in batches:
                            self.batch_queue.put(b)
                        batches = []
                    else:
                        while len(batches) >= self.aggregate:
                            b = self.aggregate_batches(batches[:self.aggregate])
                            batches = batches[self.aggregate:]
                            self.batch_queue.put(b)

            # Process remaining batches.
            if self.aggregate is None:
                for b in batches:
                    self.batch_queue.put(b)
                batches = []
            else:
                while batches:
                    b = self.aggregate_batches(batches[:self.aggregate])
                    batches = batches[self.aggregate:]
                    self.batch_queue.put(b)

            # Signal end of epoch.
            self.done_flag.set()
            self.restart_flag.wait()
            self.restart_flag.clear()

            # Distribute new files to workers
            files = list(self._rng.permutation(self.files))
            n = len(files) // self.n_workers
            if len(files) % self.n_workers > 0:
                n += 1
            for w, fs in zip(self.workers, split(files, n)):
                w.file_queue.put(fs)
                w.done_flag.clear()

    def next_epoch(self):
        """
        Sends signal to manager to start loading of next epoch.
        """
        self.done_flag.clear()
        self.restart_flag.set()

    def shutdown(self):
        """
        Sends signal to manager to start loading of next epoch.
        """
        if hasattr(self, "workers"):
            for w in self.workers:
                w.terminate()
        self.terminate()


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
        n_workers=4,
        queue_size=64,
        aggregate=None,
        shuffle=True,
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
        self.n_workers = n_workers
        self.queue_size = queue_size
        self.shuffle = shuffle
        self.aggregate = aggregate

        pool = ThreadPoolExecutor(max_workers=n_workers)
        self.folder.download(pool)
        self.files = [self.folder.get(f) for f in self.folder.files]

        self.manager = DatasetManager(self.dataset_factory,
                                      self.files,
                                      self.n_workers,
                                      self.queue_size,
                                      aggregate=self.aggregate,
                                      shuffle=self.shuffle,
                                      args=self.args,
                                      kwargs=self.kwargs)
        self.manager.daemon = True
        self.manager.start()

    def __del__(self):
        if hasattr(self, "manager"):
            self.manager.shutdown()

    def __iter__(self):
        """
        Iterate over all batches in all remote files.
        """
        # Iterate while there's batches in the manager's batch queue
        # and the manager hasn't finished processing the epoch yet.
        while ((not self.manager.done_flag.is_set())
               or self.manager.batch_queue.qsize()):
            try:
                b = self.manager.batch_queue.get_nowait()
                yield b
            except FileNotFoundError:
                _LOGGER.warning(
                    "FileNotFoundError occured when retrieving batch from "
                    "manager process."
                )
            except queue.Empty:
                continue
        self.manager.next_epoch()


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
