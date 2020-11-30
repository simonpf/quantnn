"""
quantnn.common
==============

Implements common features used by the other submodules of the ``quantnn``
package.
"""
import numpy as np
import numpy.ma as ma

BACKENDS = {"numpy": np,
            "numpy.ma.core": ma}
try:
    import torch
    BACKENDS["torch"] = torch
except ModuleNotFoundError:
    pass


class QRNNException(Exception):
    """ Base exception for exception from the QRNN package."""

class UnknownArrayTypeException(QRNNException):
    """Thrown when a function is called with an unsupported array type."""

class InvalidDimensionException(QRNNException):
    """Thrown when an input array doesn't match expected shape."""

def get_array_module(x):
    """
    Args:
        An input array or tensor object.

    Returns:
        The module object providing array operations on the
        array type.
    """
    module_name = type(x).__module__
    if module_name in BACKENDS:
        return BACKENDS[module_name]
    raise UnknownArrayTypeException(f"The provided input of type {type(x)} is not a"
                           "supported array type.")

def to_array(module, array):
    if module.__name__ == "numpy":
        return module.asarray(array)
    elif module.__name__ == "torch":
        return module.tensor(array)


def sample_uniform(module, shape):
    if module.__name__ == "numpy":
        return module.random.rand(*shape)
    elif module.__name__ == "torch":
        return module.rand(shape)

def sample_gaussian(module, shape):
    if module.__name__ in ["numpy", "numpy.ma.core"]:
        return module.random.randn(*shape)
    elif module.__name__ == "torch":
        return module.randn(*shape)
