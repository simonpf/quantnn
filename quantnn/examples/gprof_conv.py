"""
=======================
quantnn.gprof_conv.py
=======================

This module provides download functions and dataset classes for the
 convolutional GPROF retrieval example.
"""
from pathlib import Path
from urllib.request import urlretrieve


def download_data(destination="data"):
    """
    Downloads training and evaluation data for the CTP retrieval.

    Args:
        destination: Where to store the downloaded data.
    """
    datasets = [
        "gprof_conv.npz",
    ]

    Path(destination).mkdir(exist_ok=True)
    for file in datasets:
        file_path = Path("data") / file
        if not file_path.exists():
            print(f"Downloading file {file}.")
            url = f"http://spfrnd.de/data/gprof/{file}"
            urlretrieve(url, file_path)
