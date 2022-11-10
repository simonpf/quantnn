"""
quantnn.generic
===============

This module provides backend-agnostic array operations.
"""
import numpy as np
import numpy.ma as ma
import scipy as sp

from quantnn.common import (
    UnknownArrayTypeException,
    UnknownModuleException,
    InvalidDimensionException,
)

# Placeholders for modules.
torch = None
jnp = None
jax = None
_JAX_KEY = None
tf = None


def _get_backend_module(name):
    """
    Return module object corresponding to given backend.

    Args:
       The name of the backend.

    Return:
       The corresponding module object.
    """
    if name == "numpy":
        import numpy as np

        return np
    if name == "numpy.ma":
        import numpy as np

        return np.ma
    if name == "torch":
        import torch

        return torch
    if name == "jax":
        import jax
        import jax.numpy as jnp

        _JAX_KEY = jax.random.PRNGKey(0)
        return jnp
    if name == "tensorflow":
        import tensorflow as tf

        return tf


def _import_modules():
    global torch, jax, jnp, _JAX_KEY, tf

    try:
        import torch
    except ModuleNotFoundError:
        pass

    try:
        import jax
        import jax.numpy as jnp

        _JAX_KEY = jax.random.PRNGKey(0)
    except ModuleNotFoundError:
        pass

    try:
        import tensorflow as tf
    except:
        pass


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
        return _get_backend_module("numpy")
    if module_name == "numpy.ma":
        return _get_backend_module("numpy.ma")
    base_module = module_name.split(".")[0]
    if base_module == "torch":
        return _get_backend_module("torch")
    if base_module == "jax":
        return _get_backend_module("jax")
    if base_module == "tensorflow":
        return _get_backend_module("tensorflow")
    raise UnknownArrayTypeException(
        f"The provided input of type {type(x)} is" "not a supported array type."
    )


def to_array(module, array, like=None):
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
    _import_modules()
    if module in [np, ma]:
        if like is not None:
            return module.asarray(array, dtype=like.dtype)
        else:
            return module.asarray(array)
    elif module == torch:
        if isinstance(array, torch.Tensor):
            if like is None:
                return array
            else:
                t = array.to(device=like.device, dtype=like.dtype)
                if like.requires_grad:
                    t.requires_grad = True
                return t

        if like is not None:
            return module.tensor(
                array,
                dtype=like.dtype,
                device=like.device,
                requires_grad=like.requires_grad,
            )
        else:
            return module.tensor(array)
    elif module == jnp:
        return module.asarray(array)
    elif module == tf:
        return module.convert_to_tensor(array)
    raise UnknownModuleException(f"Module {module.__name__} not supported.")


def sample_uniform(module, shape, like=None):
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
    _import_modules()
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
    _import_modules()
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
    _import_modules()
    module_name = type(array).__module__.split(".")[0]
    if module_name in ["numpy", "numpy.ma.core"]:
        return array.size
    elif module_name == "torch":
        return array.numel()
    elif module_name.split(".")[0] == "jax":
        return array.size
    elif module_name.split(".")[0] == "tensorflow":
        return tf.size(array)
    raise UnknownArrayTypeException(
        f"The provided input of type {type(array)} is"
        "not a supported array type."
    )


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
    _import_modules()
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
    _import_modules()
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
    _import_modules()
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
    _import_modules()
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
    _import_modules()
    if module in [np, ma, jnp]:
        return x.astype(y.dtype)
    elif module == tf:
        return tf.cast(x, y.dtype)
    elif module == torch:
        return x.to(dtype=y.dtype, device=y.device)
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
    _import_modules()
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
    _import_modules()
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
    return module.math.reduce_sum(
        0.5 * (dx * y[selection_l] + dx * y[selection_r]), axis=dimension
    )


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
    if len(x) == y.shape[dimension] + 1:
        dx = x[1:] - x[:-1]
        n = len(y.shape)
        dx_shape = [1] * n
        dx_shape[dimension] = -1
        dx = reshape(module, dx, dx_shape)
        return module.sum(y * dx, dimension)

    if module in [np, ma, torch, jnp]:
        return module.trapz(y, x=x, axis=dimension)
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
    _import_modules()
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
    if len(x.shape) < n:
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

    if dx.shape[dimension] == y.shape[dimension]:
        y_int = cumsum(module, dx * y, dimension)
    elif dx.shape[dimension] == y.shape[dimension] - 1:
        y_int = cumsum(
            module, 0.5 * (dx * y[selection_l] + dx * y[selection_r]), dimension
        )
    else:
        raise InvalidDimensionException(
            "To integrate y over x, x must have exactly as many or one more "
            "along dimension."
        )

    return pad_zeros_left(module, y_int, 1, dimension)


def zeros(module, shape, like=None):
    """
    Zero tensor of given shape.

    Arguments:
        module: The backend array corresponding to the given array.
        shape: Tuple defining the desired shape of the tensor to create.
        like: Optional tensor to use to determine additional properties
            such as data type, device, etc ...

    Return:
         Zero tensor of given shape.
    """
    _import_modules()
    if module in [np, ma]:
        if like is not None:
            return module.zeros(shape, dtype=like.dtype)
        else:
            return module.zeros(shape)
    elif module == torch:
        if like is not None:
            return module.zeros(shape, dtype=like.dtype, device=like.device)
        else:
            return module.zeros(shape)
    elif module == jnp:
        return module.zeros(shape)
    elif module == tf:
        return module.zeros(shape)
    raise UnknownModuleException(f"Module {module.__name__} not supported.")


def ones(module, shape, like=None):
    """
    One tensor of given shape.

    Arguments:
        module: The backend array corresponding to the given array.
        shape: Tuple defining the desired shape of the tensor to create.
        like: Optional tensor to use to determine additional properties
            such as data type, device, etc ...

    Return:
         One tensor of given shape.
    """
    _import_modules()
    if module in [np, ma]:
        if like is not None:
            return module.ones(shape, dtype=like.dtype)
        else:
            return module.ones(shape)
    elif module == torch:
        if like is not None:
            return module.ones(shape, dtype=like.dtype, device=like.device)
        else:
            return module.ones(shape)
    elif module == jnp:
        return module.ones(shape)
    elif module == tf:
        return module.ones(shape)
    raise UnknownModuleException(f"Module {module.__name__} not supported.")


def softmax(module, x, axis=None):
    """
    Apply softmax to tensor.

    Arguments:
        module: The backend array corresponding to the given array.
        x: The tensor to apply the softmax to.

    Return:
         softmax(x)
    """
    _import_modules()
    if module in [np, ma]:
        return sp.special.softmax(x, axis=axis)
    elif module == torch:
        return module.nn.functional.softmax(x, dim=axis)
    elif module == jnp:
        return jax.nn.softmax(x, axis=axis)
    elif module == tf:
        return module.nn.softmax(x, axis=axis)
    raise UnknownModuleException(f"Module {module.__name__} not supported.")


def sigmoid(module, x):
    """
    Apply element-wise sigmoid to tensor.

    Arguments:
        module: The backend array corresponding to the given array.
        x: The tensor to apply the softmax to.

    Return:
         sigmoid(x)
    """
    _import_modules()
    if module in [np, ma]:
        return sp.special.sigmoid(x)
    elif module == torch:
        return module.sigmoid(x)
    elif module == jnp:
        return jax.nn.sigmoid(x)
    elif module == tf:
        return module.nn.sigmoid(x)
    raise UnknownModuleException(f"Module {module.__name__} not supported.")


def exp(module, x):
    """
    Calculate exponential of tensor.

    Arguments:
        module: The backend array corresponding to the given array.
        x: The tensor to calculate the exponential of.

    Return:
         exp(x)
    """
    _import_modules()
    if module in [np, ma]:
        return np.exp(x)
    elif module == torch:
        return torch.exp(x)
    elif module == jnp:
        return jnp.exp(x)
    elif module == tf:
        return tf.math.exp(x)
    raise UnknownModuleException(f"Module {module.__name__} not supported.")


def tensordot(module, x, y, axes):
    """
    Calculate tensor product of two tensors.

    Arguments:
        module: The backend array corresponding to the given array.
        x: The left-hand-side operand
        y: The right-hand-side operand
        axes: Integer or pair of integers describing over which axes
            to calculate the tensor product.

    Return:
        The tensor containing the tensor product of the input tensors.
    """
    _import_modules()
    if module in [np, ma]:
        return np.tensordot(x, y, axes)
    elif module == torch:
        return torch.tensordot(x, y, axes)
    elif module == jnp:
        return jnp.tensordot(x, y, axes)
    elif module == tf:
        return tf.tensordot(x, y, axes)
    raise UnknownModuleException(f"Module {module.__name__} not supported.")


def argmax(module, x, axes=None):
    """
    Get indices of maximum in tensor.

    Arguments:
        module: The backend array corresponding to the given array.
        x: The tensor to calculate the exponential of.
        axes: Tuple specifying the axes along the which compute the
            maximum.

    Return:
        Tensor containing indices of the maximum in 'x'.
    """
    return module.argmax(x, axes)


def take_along_axis(module, x, indices, axis):
    """
    Take elements along axis.
    """
    if module in [np, ma]:
        return np.take_along_axis(x, indices, axis)
    elif module == torch:
        return torch.gather(x, axis, indices)
    elif module == tf:
        return tf.gather(x, indices, axis=axis)
    elif module == jnp:
        return jnp.take_along_axis(x, indices, axis)
    raise UnknownModuleException(f"Module {module.__name__} not supported.")


def digitize(module, x, bins):
    """
    Calculate bin indices.
    """
    if module in [np, ma]:
        return np.digitize(x, bins)
    elif module == torch:
        return torch.bucketize(x, bins)
    elif module == tf:
        return tf.bucketize(x, bins)
    elif module == jnp:
        return jnp.digitize(x, bins)
    raise UnknownModuleException(f"Module {module.__name__} not supported.")


def scatter_add(module, x, indices, y, axis):
    """
    Sparse addition of values in y to given indices in x.

    Note: This operation is in-place for backend that support this.

    Args:
        module: The tensor backend module.
        x: A rank-k tensor to which to add values.
        indices: 1D Tensor containing the indices along ``axis`` into which
            to add the values from y.
        y: Tensor of rank ``k-1`` from which to add element to 'x'
        axis: Index defining the axis along which to perform the sparse
            addition.
    """
    if module in [np, ma, torch, tf]:
        selection_out = [slice(0, None)] * x.ndim
        selection_in = [slice(0, None)] * y.ndim
        for i, ind in enumerate(indices):
            if (ind >= 0) and (ind < x.shape[axis]):
                selection_out[axis] = ind
                selection_in[axis] = i
                x[tuple(selection_out)] += y[tuple(selection_in)]
        return x
    elif module == jnp:
        selection_out = [slice(0, None)] * x.ndim
        selection_in = [slice(0, None)] * y.ndim
        for i, ind in enumerate(indices):
            if (ind >= 0) and (ind < x.shape[axis]):
                selection_out[axis] = ind
                selection_in[axis] = i
                x = x.at[tuple(selection_out)].add(y[tuple(selection_in)])
        return x
    raise UnknownModuleException(f"Module {module.__name__} not supported.")

