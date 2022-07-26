import os

import numpy as np
import pytest
from quantnn.files import sftp

# Currently no SFTP test data available.
HAS_LOGIN_INFO = False

@pytest.mark.skipif(not HAS_LOGIN_INFO, reason="No SFTP login info.")
def test_list_files():
    host = "129.16.35.202"
    path = "/mnt/array1/share/MLDatasets/test"
    files = sftp.list_files(host, path)
    assert len(files) == 8

@pytest.mark.skipif(not HAS_LOGIN_INFO, reason="No SFTP login info.")
def test_download_file():
    """
    Ensure that downloading of files work and the data is cleaned up after
    usage.
    """
    host = "129.16.35.202"
    path = "/mnt/array1/share/MLDatasets/test/data_0.npz"
    tmp_file = None
    with sftp.download_file(host, path) as file:
        tmp_file = file
        data = np.load(file)
        assert np.all(np.isclose(data["x"], 0.0))

    assert not tmp_file.exists()

@pytest.mark.skipif(not HAS_LOGIN_INFO, reason="No SFTP login info.")
def test_sftp_cache():
    """
    Ensure that downloading of files work and the data is cleaned up after
    usage.
    """
    host = "129.16.35.202"
    path = "/mnt/array1/share/MLDatasets/test/data_0.npz"

    cache = sftp.SFTPCache()
    file = cache.get(host, path)
    data = np.load(file, allow_pickle=True)
    assert np.all(np.isclose(data["x"], 0.0))
