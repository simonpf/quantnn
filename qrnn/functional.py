"""
qrnn.functional
===============

The ``qrnn.functional`` module provides functions to calculate relevant
 statistics from QRNN output.
"""
from copy import copy

import numpy as np
from qrnn.common import (UnknownArrayTypeException,
                         InvalidDimensionException)

def _get_array_module(x):
    """
    Args:
        An input array or tensor object.

    Returns:
        The module object providing array operations on the
        array type.
    """
    if isinstance(x, np.ndarray):
        return np
    import torch
    if isinstance(x, torch.tensor):
        return torch
    raise UnknownArrayTypeException(f"The provided input of type {type(x)} is not a"
                           "supported array type.")

def cdf(y_pred,
        quantiles,
        quantile_axis=1):
    """
    Calculates the cumulative distribution function (CDF) from predicted
    quantiles.

    Args:
        y_pred: Array containing a range of predicted quantiles. The array
            is expected to contain the quantiles along the axis given by
            ``quantile_axis.``
        quantiles: Array containing quantile fraction corresponding to the
            the predicted quantiles.
        quantile_axis: The index of the axis f the ``y_pred`` array, along
            which the quantiles are found.

    Returns:
        Tuple ``(x_cdf, y_cdf)`` of x and corresponding y-values of the CDF
        corresponding to quantiles given by ``y_pred``.

    Raises:

        InvalidArrayTypeException: When the data is provided neither as
             numpy array nor as torch tensor.

        InvalidDimensionException: When the provided predicted quantiles do
             not match the provided number of quantiles.
    """
    if len(y_pred.shape) == 1:
        quantile_axis = 0
    if y_pred.shape[quantile_axis] != len(quantiles):
        raise InvalidDimensionException(
            "Dimensions of the provided array 'y_pred' do not match the"
            "provided number of quantiles."
        )

    output_shape = list(y_pred.shape)
    output_shape[quantile_axis] += 2
    n_dims = len(output_shape)
    xp = _get_array_module(y_pred)

    y_cdf = xp.zeros(len(quantiles) + 2)
    y_cdf[1:-1] = quantiles
    y_cdf[0] = 0.0
    y_cdf[-1] = 1.0

    x_cdf = xp.zeros(output_shape)

    selection = [slice(0, None)] * len(output_shape)
    selection[quantile_axis] = slice(1, -1)
    x_cdf[tuple(selection)] = y_pred

    selection_l = copy(selection)
    selection_l[quantile_axis] = 0
    selection_l = tuple(selection_l)
    selection_c = copy(selection)
    selection_c[quantile_axis] = 1
    selection_c = tuple(selection_c)
    selection_r = copy(selection)
    selection_r[quantile_axis] = 2
    selection_r = tuple(selection_r)
    dy = (x_cdf[selection_r] - x_cdf[selection_c]) / (quantiles[1] - quantiles[0])
    x_cdf[selection_l] = x_cdf[selection_c] - quantiles[0] * dy

    selection_l = copy(selection)
    selection_l[quantile_axis] = -3
    selection_l = tuple(selection_l)
    selection_c = copy(selection)
    selection_c[quantile_axis] = -2
    selection_c = tuple(selection_c)
    selection_r = copy(selection)
    selection_r[quantile_axis] = -1
    selection_r = tuple(selection_r)
    dy = (x_cdf[selection_c] - x_cdf[selection_l]) / (quantiles[-1] - quantiles[-2])
    x_cdf[selection_r] = x_cdf[selection_c] + (1.0 - quantiles[-1]) * dy

    return x_cdf, y_cdf


def pdf(y_pred,
        quantiles,
        quantile_axis=1):
    """
    Calculate probability density function (PDF) of the posterior distribution
    defined by predicted quantiles.

    The PDF is approximated by computing the derivative of the cumulative
    distribution function (CDF), which is obtained from by fitting a piece-wise
    function to the predicted quantiles and corresponding quantile fractions.

    Args:
        y_pred: Tensor containing the predicted quantiles along the quantile
            axis.
        quantiles: The quantile fractions corresponding to the predicted
            quantiles in y_pred.
        quantile_axis: The axis of y_pred along which the predicted
            quantiles are located.

    Returns:
        Tuple ``(x_pdf, y_pdf)`` consisting of two arrays. ``x_pdf``
        corresponds to the x-values of the PDF. ``y_pdf`` corresponds
        to the y-values of the PDF.
    """
    if len(y_pred.shape) == 1:
        quantile_axis = 0

    xp = _get_array_module(y_pred)

    x_cdf, y_cdf = cdf(y_pred, quantiles, quantile_axis=quantile_axis)
    output_shape = list(x_cdf.shape)
    output_shape[quantile_axis] += 1
    n_dims = len(output_shape)

    x_pdf = xp.zeros(output_shape)
    selection_l = [slice(0, None)] * n_dims
    selection_l[quantile_axis] = slice(1, None)
    selection_l = tuple(selection_l)
    selection_r = [slice(0, None)] * n_dims
    selection_r[quantile_axis] = slice(0, -1)
    selection_r = tuple(selection_r)
    selection_c = [slice(0, None)] * n_dims
    selection_c[quantile_axis] = slice(1, -1)
    selection_c = tuple(selection_c)
    x_pdf[selection_c] = 0.5 * (x_cdf[selection_l] + x_cdf[selection_r])

    selection = [slice(0, None)] * n_dims
    selection[quantile_axis] = 0
    selection = tuple(selection)
    x_pdf[selection] = x_cdf[selection]
    selection = [slice(0, None)] * n_dims
    selection[quantile_axis] = -1
    selection = tuple(selection)
    x_pdf[selection] = x_cdf[selection]

    y_pdf = xp.zeros(output_shape)
    y_pdf[selection_c] = np.diff(y_cdf)
    y_pdf[selection_c] /= np.diff(x_cdf, axis=quantile_axis)

    return x_pdf, y_pdf


def posterior_mean(y_pred, quantiles, quantile_axis=1):
    r"""
    Computes the mean of the posterior distribution defined by an array
    of predicted quantiles.

    Args:
        y_pred: A tensor of predicted quantiles with the quantiles located
             along the axis given by ``quantile_axis``.
        quantiles: The quantile fractions corresponding to the quantiles
             located along the quantile axis.
        quantile_axis: The axis along which the quantiles are located.

    Returns:

        Array containing the posterior means for the provided inputs.
    """
    if len(y_pred.shape) == 1:
        quantile_axis = 0
    xp = _get_array_module(y_pred)

    x_cdf, y_cdf = cdf(y_pred, quantiles, quantile_axis=quantile_axis)
    return xp.trapz(x_cdf, x=y_cdf, axis=quantile_axis)
