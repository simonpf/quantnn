from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
from pathlib import Path
import tempfile

import numpy as np
import paramiko
from quantnn.common import MissingAuthenticationInfo, DatasetError
from torch.utils.data import get_worker_info, IterableDataset

_DATASET_LOCK = multiprocessing.Lock()


def get_login_info():
    """
    Retrieves SFTP login info from the 'QUANTNN_SFTP_USER' AND
    'QUANTNN_SFTP_PASSWORD' environment variables.

    Returns:

        Tuple ``(user_name, password)`` containing the SFTP user name and
        password retrieved from the environment variables.

    Raises:

        MissingAuthenticationInfo exception when required information is
        not provided as environment variable.
    """
    user_name = os.environ.get("QUANTNN_SFTP_USER")
    password = os.environ.get("QUANTNN_SFTP_PASSWORD")
    if user_name is None or password is None:
        raise MissingAuthenticationInfo(
            "SFTPStream dataset requires the 'QUANTNN_SFTP' and "
            "'QUANTNN_SFTP_PASSWORD' to be set."
        )
    return user_name, password

@contextmanager
def get_sftp_connection(host):
    """
    Contextmanager to open and close an SFTP connection to
    a given host.

    Login credentials for the SFTP server are retrieved from the
    'QUANTNN_SFTP_USER' and 'QUANTNN_SFTP_PASSWORD' environment variables.

    Args:
        host: IP address of the host.

    Returns:
        ``paramiko.SFTP`` object providing access to the open SFTP connection.
    """
    user_name, password = get_login_info()
    transport = None
    sftp = None
    try:
        transport = paramiko.Transport(host)
        transport.connect(username=user_name,
                          password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        yield sftp
    finally:
        if sftp:
            sftp.close()
        if transport:
            transport.close()

def list_files(host, path):
    """
    List files in SFTP folder.

    Args:
        host: IP address of the host.
        path: The path for which to list the files


    Returns:
        List of absolute paths to the files discovered under
        the given path.
    """
    with get_sftp_connection(host) as sftp:
        files = sftp.listdir(path)
    return [Path(path) / f for f in files]

def iterate_dataset(dataset):
    """
    Turns an iterable or sequence dataset into a generator.

    Returns:

        Generator object providing access to the batches in the dataset.

    Raises:

        quantnn.DatasetError when the dataset is neither iterable nor
        a sequence.

    """
    if isinstance(dataset, Iterable):
        yield from dataset
    elif hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        for i in range(len(dataset)):
            yield dataset[i]
    else:
        raise DatasetError("The provided dataset is neither iterable nor a sequence.")

def open_dataset(host,
                 path,
                 dataset_factory,
                 args=None,
                 kwargs=None):
    """
    Downloads file using SFTP and opens dataset using a temporary directory
    for file transfer.

    Args:
        host: IP address of the host.
        path: The path for which to list the files
        dataset_class: The class used to read in the file.
        args: List of positional arguments passed to the dataset_factory method
              after the downloaded file.
        kwargs: Dictionary of keyword arguments passed to the dataset
            factory call.

    Returns:
        An object created using the provided dataset_factory
        using the downloaded file as first arguments and the provided
        args and kwargs as positional and keyword arguments.
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

    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / path.name
        with get_sftp_connection(host) as sftp:
            sftp.get(str(path), str(destination))
            with _DATASET_LOCK:
                dataset = dataset_factory(destination, *args, **kwargs)
    return dataset


class SFTPStream(IterableDataset):
    def __init__(self,
                 host,
                 path,
                 dataset_factory,
                 args=None,
                 kwargs=None):
        self.host = host
        self.path = path
        self.dataset_factory = dataset_factory
        self.args = args
        self.kwargs = kwargs
        self.files = list_files(self.host, self.path)
        # Sort datasets into random order.
        self.files = np.random.permutation(self.files)



    def __iter__(self):

        pool = ThreadPoolExecutor(max_workers=2)

        info = get_worker_info()
        if info is None:
            n_workers = 1
            worker_id = 0
        else:
            n_workers = info.num_workers
            worker_id = info.id

        indices = np.arange(worker_id, len(self.files), n_workers)

        if len(indices) == 0:
            raise StopIteration

        files = [self.files[i] for i in indices]

        datasets = []
        arguments = [self.host, files[0], self.dataset_factory,
                     self.args, self.kwargs]
        datasets.append(pool.submit(open_dataset, *arguments))

        for i in range(len(files)):
            if i + 1 < len(files):
                arguments = [self.host, files[i+1], self.dataset_factory,
                             self.args, self.kwargs]
                datasets.append(pool.submit(open_dataset, *arguments))
            dataset = datasets.pop(0).result()
            yield from iterate_dataset(dataset)


    def _open_connection(self, username, password):
        transport = paramiko.Transport(self.host)
        transport.connect(username=username,
                          password=password)
        self.sftp = paramiko.SFTPClient.from_transport(transport)

    def _discover_files(self):
        self.sftp.chdir(self.path)
        self.files = self.sftp.listdir()


os.environ["QUANTNN_SFTP_USER"] = "simon"
os.environ["QUANTNN_SFTP_PASSWORD"] = "dendrite_geheim"

stream = SFTPStream(
    "129.16.35.202",
    "array1/share/Datasets/gprof/simple/training_data",
    str
)

