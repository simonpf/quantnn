"""
==================
quantnn.files.sftp
==================

This module provides high-level functions to access file via
SFTP.
"""
from contextlib import contextmanager
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
