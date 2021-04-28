"""
=============
quantnn.utils
=============

This module providers Helper functions that are used in multiple other modules.
"""
import io
from pathlib import Path
from tempfile import mkstemp

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
    if all([isinstance(x, dict) for x in args]):
        return {
            k: f(*[x[k] for x in args]) for k in args[0]
        }
    return f(*args)

def serialize_dataset(dataset):
    """
    Writes xarray dataset to a bytestream.

    Args:
         dataset: A xarray dataset to seraialize.

    Returns:
         Bytes object containing the dataset as netcdf file.
    """
    _, filename = mkstemp()
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

    _, filename = mkstemp()
    try:
        with open(filename, "wb") as file:
            buffer = file.write(data)
        dataset = xr.load_dataset(filename)
    finally:
        Path(filename).unlink()
    return dataset
