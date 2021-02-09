"""
==================
quantnn.files.sftp
==================

This module provides high-level functions to access file via
SFTP.
"""
from contextlib import contextmanager
import io
import os
from pathlib import Path
import tempfile

import paramiko
from quantnn.common import MissingAuthenticationInfo, DatasetError



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
            sftp.get(str(path), str(destination))
            yield destination


class TemporaryFile:
    """
    Generic temporary file that can be either on disk or in memory.

    Attributes:
        file: tempfile.TemporaryFile object or io.BytesIO.
    """
    def __init__(self, on_disk=True):
        """
        Create new temporary file.

        Args:
           on_disk: Whether the data should be stored on disk (True)
               or in memory (False).
        """
        self.on_disk = on_disk
        if self.on_disk:
            self.file = tempfile.TemporaryFile()
        else:
            self.file = io.BytesIO()
        self.closed = False

    def close(self):
        """ Forwarded to file attribute. """
        if self.on_disk and not self.closed:
            self.file_close()
            self.closed = False

    def read(self, *args, **kwargs):
        """ Forwarded to file attribute. """
        return self.file.read(*args, **kwargs)

    def readlines(self, *args, **kwargs):
        """ Forwarded to file attribute. """
        return self.file.readlines(*args, **kwargs)

    def seek(self, *args, **kwargs):
        """ Forwarded to file attribute. """
        return self.file.seek(*args, **kwargs)

    def file_close(self, *args, **kwargs):
        """ Forwarded to file attribute. """
        return self.file.file_close(*args, **kwargs)

    def flush(self):
        """ Forwarded to file attribute. """
        return self.file.flush()

    @property
    def seekable(self):
        """ Forwarded to file attribute. """
        return self.file.seekable

    def tell(self, *args, **kwargs):
        """ Forwarded to file attribute. """
        return self.file.tell(*args, **kwargs)

    def write(self, *args, **kwargs):
        """ Forwarded to file attribute. """
        self.file.write(*args, **kwargs)

    def __del__(self):
        self.close()

class SFTPCache:
    """
    Cache for SFTP files.


    Attributes:
        files: Dictionary mapping tuples ``(host, path)`` to temporary
            file object.
    """
    def __init__(self, on_disk=True):
        self.on_disk = on_disk
        self.files = {}


    def get(self, host, path):
        """
        Retrieve file from cache. If file is not found in cache it is
        retrieved via SFTP and stored in the cache.

        Args:
            host: The SFTP host from which to retrieve the file.
            path: The path of the file on the host.

        Return:
            The requested file.
        """
        key = (host, path)
        if not key in self.files:
            file = TemporaryFile(self.on_disk)
            with get_sftp_connection(host) as sftp:
                sftp.getfo(str(path), file)
                file.seek(0)
            self.files[key] = file
        return self.files[key]
