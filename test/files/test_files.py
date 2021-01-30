"""
Test for the generic function in the :py:mod:`quantnn.files module`.
"""
import os

import pytest
import numpy as np
from quantnn.files import read_file

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

@pytest.mark.skipif(not HAS_LOGIN_INFO, reason="No SFTP login info.")
def test_remote_file():
    """
    Ensures that opening a local file works.
    """
    with read_file("sftp://129.16.35.202/mnt/array1/share/Datasets/test/data_0.npz", "rb") as file:
        data = np.load(file)

    assert np.all(np.isclose(data["x"], 0.0))
