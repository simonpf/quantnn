"""
quantnn.generic
===============

This module provides backend-agnostic array operations.
"""
import numpy as np
import numpy.ma as ma
import itertools

from quantnn.common import (UnknownArrayTypeException,
                            UnknownModuleException)

BACKENDS = {"numpy": np,
            "numpy.ma.core": ma,
            }
try:
    import torch
    BACKENDS["torch"] = torch
except ModuleNotFoundError:
    torch = None
    pass
try:
    import jax
    import jax.numpy as jnp
    _JAX_KEY = jax.random.PRNGKey(0)
    BACKENDS["jax"] = jnp
except ModuleNotFoundError:
    jnp = None


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
        return jax.random.uniform(_JAX_KEY, shape)
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
        return jax.random.normal(_JAX_KEY, shape)
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
    elif module_name.split(".")[0] == "jax":
        return array.size
    raise UnknownArrayTypeException(f"The provided input of type {type(x)} is"
                                    "not a supported array type.")


def concatenate(module, arrays, dimension):
    """
    Concatenate array along given dimension.

    Args:
        module: Module object corresponding to the arrays.
        arrays: List of arrays to concatenate.
        dimension: Index of the dimensions along which to concatenate.

    Return:
        The array resulting from concatenating the given arrays along
        the given dimension.
    """
    if module in [np, ma, jnp]:
        return module.concatenate(arrays, dimension)
    elif module == torch:
        return module.cat(arrays, dimension)
    return UnknownModuleException(f"Module {module.__name__} not supported.")


def expand_dims(module, array, dimension):
    """
    Expand tensor dimension along given axis.

    Inserts a dimension of length one at a given index of the
    dimension array.

    Args:
        module: Module object corresponding to the arrays.
        array: The array whose dimension to expand.
        dimension: The index at which to insert the new
            dimension.

    Returns:
        The reshaped array with a dimension added at the given index.
    """
    if module in [np, ma, jnp]:
        return module.expand_dims(array, dimension)
    elif module == torch:
        return module.unsqueeze(array, dimension)
    return UnknownModuleException(f"Module {module.__name__} not supported.")

def pad_zeros(module, array, n, dimension):
    """
    Pads array with 0s along given dimension.

    Args:
        module: Module object corresponding to the arrays.
        array: The array to pad.
        n: The number of zeros to add to each edge.
        dimension: Along which dimension to add zeros.

    Returns:
        A new array with the given number of 0s added to
        each edge along the given dimension.
    """
    if module in [np, ma, jnp]:
        n_dims = len(array.shape)
        pad = [(0, 0)] * n_dims
        pad[dimension] = (n, n)
        return module.pad(array, pad, mode="constant", constant_values=0.0)
    elif module == torch:
        n_dims = len(array.shape)
        dimension = dimension % n_dims
        pad = [0] * 2 * n_dims
        pad[2 * n_dims - 2 - 2 * dimension] = n
        pad[2 * n_dims - 1 - 2 * dimension] = n
        return module.nn.functional.pad(array, pad, "constant", 0.0)
    return UnknownModuleException(f"Module {module.__name__} not supported.")
