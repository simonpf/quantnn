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
    BACKENDS["jax.numpy"] = jnp
except ModuleNotFoundError:
    jnp = None

try:
    import tensorflow as tf
    BACKENDS["tensorflow"] = tf
except ModuleNotFoundError:
    tf = None


def get_array_module(x):
    """
    Args:
        An input array or tensor object.

    Returns:
        The module object providing array operations on the
        array type.
    """
    module_name = type(x).__module__
    if module_name == "numpy":
        return np
    if module_name == "numpy.ma":
        return ma
    base_module = module_name.split(".")[0]
    if base_module == "torch":
        return torch
    if base_module == "jax":
        return jnp
    if base_module == "tensorflow":
        return tf
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
    if module in [np, ma]:
        return module.asarray(array)
    elif module == torch:
        return module.tensor(array)
    elif module == jnp:
        return module.asarray(array)
    elif module == tf:
        return module.convert_to_tensor(array)
    raise UnknownModuleException(f"Module {module.__name__} not supported.")


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
    if module in [np, ma]:
        return module.random.rand(*shape)
    elif module == torch:
        return module.rand(shape)
    elif module == jnp:
        return jax.random.uniform(_JAX_KEY, shape)
    elif module == tf:
        return tf.random.uniform(shape)

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
    if module in [np, ma]:
        return module.random.randn(*shape)
    elif module == torch:
        return module.randn(*shape)
    elif module == jnp:
        return jax.random.normal(_JAX_KEY, shape)
    elif module == tf:
        return tf.random.normal(shape)
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
    elif module_name.split(".")[0] == "tensorflow":
        return tf.size(array)
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
    elif module == tf:
        return tf.concat(arrays, axis=dimension)
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
    if module in [np, ma, jnp, tf]:
        return module.expand_dims(array, dimension)
    elif module == torch:
        return module.unsqueeze(array, dimension)
    raise UnknownModuleException(f"Module {module.__name__} not supported.")


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
    if module in [np, ma, jnp, tf]:
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
    raise UnknownModuleException(f"Module {module.__name__} not supported.")

def pad_zeros_left(module, array, n, dimension):
    """
    Pads array with 0s along given dimension but only on left side.

    Args:
        module: Module object corresponding to the arrays.
        array: The array to pad.
        n: The number of zeros to add to each edge.
        dimension: Along which dimension to add zeros.

    Returns:
        A new array with the given number of 0s added to
        only the left edge along the given dimension.
    """
    if module in [np, ma, jnp, tf]:
        n_dims = len(array.shape)
        pad = [(0, 0)] * n_dims
        pad[dimension] = (n, 0)
        return module.pad(array, pad, mode="constant", constant_values=0.0)
    elif module == torch:
        n_dims = len(array.shape)
        dimension = dimension % n_dims
        pad = [0] * 2 * n_dims
        pad[2 * n_dims - 2 - 2 * dimension] = n
        pad[2 * n_dims - 1 - 2 * dimension] = 0
        return module.nn.functional.pad(array, pad, "constant", 0.0)
    raise UnknownModuleException(f"Module {module.__name__} not supported.")


def as_type(module, x, y):
    """
    Converts data type of input ``x`` to that of input ``y``.

    Arguments:
         x: The array to be converted
         y: The array to whose data type to convert x.

    Return:
         The input ``x`` converted to the data type of ``y``.
    """
    if module in [np, ma, jnp]:
        return x.astype(y.dtype)
    elif module == tf:
        return tf.cast(x, y.dtype)
    elif module == torch:
        return x.to(y.dtype)
    raise UnknownModuleException(f"Module {module.__name__} not supported.")


def arange(module, start, end, step):
    """
    Crate array with stepped sequence of values.

    Arguments:
        module: The backend array corresponding to the given array.
        start: Start value of the sequence.
        end: Maximum value of the sequence.
        step: Step size.

    Return:
        1D array containing the sequence starting with the given start value
        increasing with the given step size up until the largest value that
        is strictly smaller than the given end value.
    """
    if module in [np, ma, jnp]:
        return module.arange(start, end, step)
    elif module == torch:
        return module.arange(start, end, step, dtype=torch.float)
    elif module == tf:
        return tf.range(start, end, step)
    raise UnknownModuleException(f"Module {module.__name__} not supported.")


def reshape(module, array, shape):
    """
    Reshape array into given shape.

    Arguments:
        module: The backend array corresponding to the given array.
        array: The array to reshape
        shape: The shape into which to rehshape the array.

    Returns:
        The array reshaped into the requested shape.
    """
    if module in [np, ma, torch, jnp]:
        return array.reshape(shape)
    if module == tf:
        return tf.reshape(array, shape)
    raise UnknownModuleException(f"Module {module.__name__} not supported.")


def _trapz(module, y, x, dimension):
    """
    Numeric integration using  trapezoidal rule.

    Arguments:
        module: The backend array corresponding to the given array.
        y: Rank k-tensor to integrate over the given dimension.
        x: The domain values to integrate over.
        dimension: The dimension to integrate over.

    Return:
       The rank k-1 tensor containing the numerical integrals corresponding
       to y.
    """
    n = len(y.shape)
    x_shape = [1] * n
    x_shape[dimension] = -1
    x = reshape(module, x, x_shape)

    selection = [slice(0, None)] * n
    selection_l = selection[:]
    selection_l[dimension] = slice(0, -1)
    selection_r = selection[:]
    selection_r[dimension] = slice(1, None)

    dx = x[selection_r] - x[selection_l]
    return module.math.reduce_sum(0.5 * (dx * y[selection_l] + dx * y[selection_r]), axis=dimension)


def trapz(module, y, x, dimension):
    """
    Numeric integration using  trapezoidal rule.

    Arguments:
        module: The backend array corresponding to the given array.
        y: Rank k-tensor to integrate over the given dimension.
        x: The domain values to integrate over.
        dimension: The dimension to integrate over.

    Return:
       The rank k-1 tensor containing the numerical integrals corresponding
       to y.
    """
    if module in [np, ma, torch, jnp]:
        return module.trapz(y, x=x,  axis=dimension)
    elif module == tf:
        return _trapz(module, y, x, dimension)
    raise UnknownModuleException(f"Module {module.__name__} not supported.")

def cumsum(module, y, dimension):
    """
    Cumulative sum along given axis.

    Arguments:
        module: The backend array corresponding to the given array.
        y: Rank k-tensor to accumulate along given dimension.
        dimension: The dimension to sum over.

    Return:
       The rank k tensor containing the cumulative sum along the given dimension.
    """
    if module in [np, ma, torch, jnp]:
        return module.cumsum(y, axis=dimension)
    elif module == tf:
        return tf.math.cumsum(y, dimension)

def cumtrapz(module, y, x, dimension):
    """
    Cumulative integral along given axis.

    The returned tensor has the same shape as the input tensor y and the
    values correspond to the numeric integral computed up to the corresponding
    value of the provided x vector assuming that the function described by y
    is 0 outside of the domain described by x.

    Arguments:
        module: The backend array corresponding to the given array.
        y: Rank k-tensor to accumulate along given dimension.
        dimension: The dimension to sum over.

    Return:
       The rank k tensor containing the cumulative integral along the
       given dimension.
    """
    n = len(y.shape)
    x_shape = [1] * n
    x_shape[dimension] = -1
    x = reshape(module, x, x_shape)

    selection = [slice(0, None)] * n
    selection_l = selection[:]
    selection_l[dimension] = slice(0, -1)
    selection_l = tuple(selection_l)
    selection_r = selection[:]
    selection_r[dimension] = slice(1, None)
    selection_r = tuple(selection_r)

    dx = x[selection_r] - x[selection_l]
    y_int = cumsum(module, 0.5 * (dx * y[selection_l] + dx * y[selection_r]), dimension)
    return pad_zeros_left(module, y_int, 1, dimension)


