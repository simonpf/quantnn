"""
quantnn.generic
===============

This module provides backend-agnostic array operations.
"""
import numpy as np
import numpy.ma as ma

from quantnn.common import (UnknownArrayTypeException,
                            UnknownModuleException)

BACKENDS = {"numpy": np,
            "numpy.ma.core": ma,
            }
try:
    import torch
    BACKENDS["torch"] = torch
except ModuleNotFoundError:
    pass
try:
    import jax.numpy as jnp
    BACKENDS["jax"] = jnp
except ModuleNotFoundError:
    pass


def get_array_module(x):
    """
    Args:
        An input array or tensor object.

    Returns:
        The module object providing array operations on the
        array type.
    """
    module_name = type(x).__module__.split(".")[0]
    if module_name in BACKENDS:
        return BACKENDS[module_name]
    raise UnknownArrayTypeException(f"The provided input of type {type(x)} is"
                                    "not a supported array type.")


def to_array(module, array):
    """
    Turn a list into an array.

    Args:
         module: Module representing the module which should be used as
             backend.
         array: Iterable to turn into array.

    Returns:
        Array-object corresponding to the given backend module containing the
        data in array.
    """
    if module.__name__ == "numpy":
        return module.asarray(array)
    elif module.__name__ == "torch":
        return module.tensor(array)
    elif module.__name__.split(".")[0] == "jax":
        return module.asarray(array)
    return UnknownModuleException(f"Module {module.__name__} not supported.")


def sample_uniform(module, shape):
    """
    Create a tensor with random values sampled from a uniform distribution.

    Args:
         module: Module representing the module which should be used as
             backend.
         shape: Iterable describing the shape of tensor.

    Returns:
         Array object corresponding to the given module object containing
         random values.
    """
    if module.__name__ == "numpy":
        return module.random.rand(*shape)
    elif module.__name__ == "torch":
        return module.rand(shape)
    elif module.__name__.split(".")[0] == "jax":
        return module.random.rand(*shape)
    return UnknownModuleException(f"Module {module.__name__} not supported.")


def sample_gaussian(module, shape):
    """
    Create a tensor with random values sampled from a Gaussian distribution.

    Args:
         module: Module representing the module which should be used as
             backend.
         shape: Iterable describing the shape of tensor.

    Returns:
         Array object corresponding to the given module object containing
         random values.
    """
    if module.__name__ in ["numpy", "numpy.ma.core"]:
        return module.random.randn(*shape)
    elif module.__name__ == "torch":
        return module.randn(*shape)
    elif module.__name__.split(".")[0] == "jax":
        return module.random.randn(*shape)
    return UnknownModuleException(f"Module {module.__name__} not supported.")

def numel(array):
    """
    Returns the number of elements in an array.

    Args:
         module: Module representing the module which should be used as
             backend.
         shape: Iterable describing the shape of tensor.

    Returns:
         Array object corresponding to the given module object containing
         random values.
    """
    module_name = type(array).__module__.split(".")[0]
    if module_name in ["numpy", "numpy.ma.core"]:
        return array.size
    elif module_name == "torch":
        return array.numel()
    elif module.__name__.split(".")[0] == "jax":
        return array.size
    raise UnknownArrayTypeException(f"The provided input of type {type(x)} is"
                                    "not a supported array type.")

