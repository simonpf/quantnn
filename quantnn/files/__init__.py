"""
=============
quantnn.files
=============

The :py:mod:`quantnn.files` module provides an abstraction layer to open files
locally or via SFTP.

Refer to the documentation of the :py:mod:`quantnn.files.sftp`` module for
information on how to set the username and password for the SFTP connection.

Example
-------

.. code-block::

   # To load a local file:
   with open_file("my_file.txt") as file:
        content = file.read()

   # To open a remote file via SFTP:
   with open_file("sftp://129.16.35.202/my_file.txt")


"""
from contextlib import contextmanager
from pathlib import PurePath
from urllib.parse import urlparse

from quantnn.files import sftp
from quantnn.common import InvalidURL


@contextmanager
def read_file(path, *args, **kwargs):
    """
    Generic function to open files. Currently supports opening files
    on the local system as well as on a remote machine via SFTP.
    """
    if isinstance(path, PurePath):
        yield open(path, *args, **kwargs)
        return

    url = urlparse(path)
    if url.netloc == "":
        yield open(path, *args, **kwargs)
        return

    if url.scheme == "sftp":
        host = url.netloc
        if host == "":
            raise InvalidURL(
                f"No host in SFTP URL."
                f"To load a file using SFTP, the URL must be of the form "
                f"'sftp://<host>/<path>'."
            )

        with sftp.download_file(host, url.path) as file:
            yield open(file, *args, **kwargs)
        return

    raise InvalidURL(f"The provided protocol '{url.scheme}' is not supported.")
