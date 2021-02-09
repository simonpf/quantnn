"""
==================
quantnn.files.sftp
==================

This module provides high-level functions to access file via
SFTP.
"""
from contextlib import contextmanager
from concurrent.futures import Future
import io
import logging
import os
from pathlib import Path
import tempfile

import paramiko
from quantnn.common import MissingAuthenticationInfo, DatasetError

_LOGGER = logging.getLogger("quantnn.files.sftp")


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


@contextmanager
def download_file(host,
                  path):
    """
    Downloads file from host to a temporary directory and
    return the path of this file.

    Args:
        host: IP address of the host from which to download the file.
        path: Path of the file on the host.

    Return:
        pathlib.Path object pointing to the downloaded file.
    """
    path = Path(path)
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / path.name
        with get_sftp_connection(host) as sftp:
            _LOGGER.info("Downloading file %s to %s.", path, destination)
            sftp.get(str(path), str(destination))
            yield destination


def _download_file(host, path):
    _, file = tempfile.mkstemp()
    with get_sftp_connection(host) as sftp:
        sftp.get(str(path), file)
        _LOGGER.info("Downloading file %s to %s.", path, file)
    return file

class SFTPCache:
    """
    Cache for SFTP files.


    Attributes:
        files: Dictionary mapping tuples ``(host, path)`` to temporary
            file object.
    """
    def __init__(self):
        self.files = {}

    def download_files(self, host, paths, pool):
        tasks = {}
        for path in paths:
            if (host, path) not in self.files:
                task = pool.submit(_download_file, host, path)
                tasks[path] = task
        for path in paths:
            self.files[(host, path)] = tasks[path].result()

    def cleanup(self):
        """
        Clean up temporary files.
        """
        _LOGGER.info("Cleaning up SFTP cache.")
        for file in self.files.values():
            if isinstance(file, Future):
                file = file.result()
            if type(file) is str:
                os.remove(file)
            else:
                os.remove(file.name)

    def get(self, host, path):
        """
        Retrieve file from cache. If file is not found in cache it is
        retrieved via SFTP and stored in the cache.

        Args:
            host: The SFTP host from which to retrieve the file.
            path: The path of the file on the host.

        Return:
            The temporary file object containing the requested file.
        """
        key = (host, path)
        if not key in self.files:
            _, file = tempfile.mkstemp()
            with get_sftp_connection(host) as sftp:
                sftp.getfo(str(path), file)
                file.seek(0)
            self.files[key] = file

        value = self.files[key]
        if isinstance(value, Future):
            return value.result()
        return self.files[key]
