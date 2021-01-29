import os

import pytest
from quantnn.files import sftp

HAS_LOGIN_INFO = ("QUANTNN_SFTP_USER" in os.environ and
                  "QUANTNN_SFTP_PASSWORD" in os.environ)

@pytest.mark.skipif(not HAS_LOGIN_INFO, reason="No SFTP login info.")
def test_list_files():
    host = "129.16.35.202"
    path = "/mnt/array1/share/Datasets/test"
    files = sftp.list_files(host, path)
    assert len(files) == 8

def test_download_file(tmp_path):
    """
    Ensure that downloading of files work and the data is cleaned up after
    usage.
    """
    host = "129.16.35.202"
    path = "/mnt/array1/share/Datasets/test/data_0.npz"
    tmp_file = None
    with sftp.download_file(host, path) as file:
        tmp_file = file
        data = np.load(file)
        assert np.all(np.isclose(data["x"], 0.0))

    assert not tmp_file.exists()
