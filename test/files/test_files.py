"""
Test for the generic function in the :py:mod:`quantnn.files module`.
"""
import os

import pytest
import numpy as np
from quantnn.files import read_file, CachedDataFolder
from concurrent.futures import ThreadPoolExecutor

HAS_LOGIN_INFO = ("QUANTNN_SFTP_USER" in os.environ and
                  "QUANTNN_SFTP_PASSWORD" in os.environ)

def test_local_file(tmp_path):
    """
    Ensures that opening a local file works.
    """
    with open(tmp_path / "test.txt", "w") as file:
        file.write("test")

    with read_file(tmp_path / "test.txt") as file:
        content = file.read()

    assert content == "test"


def test_local_folder(tmp_path):
    """
    Ensures that opening a local file works.
    """
    with open(tmp_path / "test.txt", "w") as file:
        file.write("test")

    data_folder = CachedDataFolder(tmp_path, "*.txt")

    assert len(data_folder.files) == 1

    file = data_folder.open(data_folder.files[0])
    assert file.read() == "test"


@pytest.mark.skipif(not HAS_LOGIN_INFO, reason="No SFTP login info.")
def test_remote_file():
    """
    Ensures that opening a local file works.
    """
    host = "129.16.35.202"
    path = "/mnt/array1/share/MLDatasets/test/data_0.npz"
    with read_file("sftp://" + host + path, "rb") as file:
        data = np.load(file)

    assert np.all(np.isclose(data["x"], 0.0))


@pytest.mark.skipif(not HAS_LOGIN_INFO, reason="No SFTP login info.")
def test_remote_folder(tmp_path):
    """
    Ensures that opening a local file works.
    """
    host = "129.16.35.202"
    path = "/mnt/array1/share/MLDatasets/test/"

    data_folder = CachedDataFolder("sftp://" + host + path, "*.npz")

    assert len(data_folder.files) == 8
    data_folder.files.sort()

    file = data_folder.get(data_folder.files[0])
    data = np.load(file)
    assert np.all(np.isclose(data["x"], 0.0))

    pool = ThreadPoolExecutor(max_workers=4)
    data_folder.download(pool)

    file = data_folder.get(data_folder.files[0])
    data = np.load(file)
    assert np.all(np.isclose(data["x"], 0.0))


