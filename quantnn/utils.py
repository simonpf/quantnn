"""
=============
quantnn.utils
=============

This module providers Helper functions that are used in multiple other modules.
"""
import io
from pathlib import Path
from tempfile import NamedTemporaryFile

import xarray as xr


def apply(f, *args):
    """
    Applies a function to sequence values or dicts of values.

    Args:
        f: The function to apply to ``x`` or all items in ``x``.
        *args: Sequence of arguments to be supplied to ``f``. If all arguments
            are dicts, the function ``f`` is applied key-wise to all elements
            in the dict. Otherwise the function is applied to all provided
            argument.s

    Return:
        ``{k: f(x_1[k], x_1[k], ...) for k in x}`` or ``f(x)`` depending on
        whether ``x_1, ...`` are a dicts or not.
    """
    if any(isinstance(x, dict) for x in args):
        results = {}
        d = [x for x in args if isinstance(x, dict)][0]
        for k in d:
            args_k = [arg[k] if isinstance(arg, dict) else arg
                      for arg in args]
            results[k] = f(*args_k)
        return results
    return f(*args)


def serialize_dataset(dataset):
    """
    Writes xarray dataset to a bytestream.

    Args:
         dataset: A xarray dataset to seraialize.

    Returns:
         Bytes object containing the dataset as netcdf file.
    """
    tmp = NamedTemporaryFile(delete=False)
    tmp.close()
    filename = tmp.name
    try:
        dataset.to_netcdf(filename)
        with open(filename, "rb") as file:
            buffer = file.read()
    finally:
        Path(filename).unlink()
    return buffer


def deserialize_dataset(data):
    """
    Read xarray dataset from byte stream containing the
    dataset in NetCDF format.

    Args:
        data: The bytes object containing the binary data of the
            NetCDf file.

    Returns:
        The deserialized xarray dataset.
    """
    tmp = NamedTemporaryFile(delete=False)
    tmp.close()
    filename = tmp.name
    try:
        with open(filename, "wb") as file:
            buffer = file.write(data)
        dataset = xr.load_dataset(filename)
    finally:
        Path(filename).unlink()
    return dataset
