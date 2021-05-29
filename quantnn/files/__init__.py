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
from pathlib import PurePath, Path
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


class _DummyCache:
    """
    A dummy cache for local files, which are not cached
    at all.

    """

    def __init__(self):
        """Create dummy cache."""
        pass

    def download_files(self, host, files, pool):
        pass

    def get(self, host, path):
        """Get file from cache."""
        return path

    def cleanup(self):
        pass


class CachedDataFolder:
    """
    This class provides an interface to a generic folder containing
    dataset files. This folder can be accessed via the local file
    system or SFTP. If the folder is located on a remote SFTP server,
    the files are cached to avoid having to retransfer the files.

    Attributes:
        files: List of available files in the folder.
        host: The name of the host where the folder is located or
             "" if the folder is local.
        cache: Cache object used to cache data accesses.
    """

    def __init__(self, path, pattern="*", n_files=None):
        """
        Create a CachedDataFolder.

        Args:
            path: Path to the folder to load.
            pattern: Glob pattern to select the files.
            n_files: If given only the first ``n_files`` matching files will
                be loaded.
        """
        if isinstance(path, PurePath):
            files = path.iterdir()
            self.host = ""
            self.cache = _DummyCache()
        else:
            url = urlparse(path)
            if url.netloc == "":
                files = Path(path).iterdir()
                self.host = ""
                self.cache = _DummyCache()
            else:
                if url.scheme == "sftp":
                    self.host = url.netloc
                    if self.host == "":
                        raise InvalidURL(
                            f"No host in SFTP URL."
                            f"To load a file using SFTP, the URL must be of the "
                            f" form 'sftp://<host>/<path>'."
                        )
                    files = sftp.list_files(self.host, url.path)
                    self.cache = sftp.SFTPCache()
                else:
                    raise InvalidURL(
                        f"The provided protocol '{url.scheme}' " f" is not supported."
                    )
        self.files = list(filter(lambda f: f.match(pattern), files))
        if n_files:
            self.files = self.files[:n_files]

    def download(self, pool):
        """
        This method downloads all files in the folder to populate the
        cache.

        Args:
            The PoolExecutor to use for the conurrent download.
        """
        self.cache.download_files(self.host, self.files, pool)

    def get(self, path):
        """
        Retrieve file from folder.

        Args:
             path: The path of the file to retrieve.

        Return:
             If it is a local file, the filename of the file is returned.
             If the file is remote a cached temporary file object with
             the data is returned.
        """
        return self.cache.get(self.host, path)

    def open(self, path, *args, **kwargs):
        """
        Retrieve file from cache and open.

        Args:
            path: The path of the file to retrieve.
            *args: Passed to open call if file is local.
            **kwargs: Passed to open call if file is local.

        """
        file = self.get(path)
        if isinstance(file, PurePath):
            return open(file, *args, **kwargs)
        return file
