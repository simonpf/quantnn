"""
===============
quantnn.density
===============

This module provides generic functions to manipulate and derive statistics
from probabilistic predictions in the form of discretized probability density
functions as produced by DRNNs.
"""
from quantnn.common import InvalidDimensionException
from quantnn.generic import (get_array_module,
                             numel,
                             expand_dims,
                             concatenate,
                             trapz,
                             cumtrapz,
                             cumsum,
                             reshape,
                             pad_zeros_left,
                             as_type)

def _check_dimensions(n_y, n_b):
    if n_y != n_b - 1:
        raise InvalidDimensionException(
            f"Dimensions of the provided array 'y_pred' ({n_y}) do not match the"
            f" provided number of bin edges ({n_b}). Note that there should be"
            f" one more bin edge than bins values is 'y_pred'."
        )

def normalize(y_pred,
              bins,
              bin_axis=1):
    if len(y_pred.shape) == 1:
        bin_axis = 0
    n_y = y_pred.shape[bin_axis]
    n_b = len(bins)
    _check_dimensions(n_y, n_b)
    xp = get_array_module(y_pred)
    n = len(y_pred.shape)

    norm = y_pred.sum(bin_axis)
    norm = expand_dims(xp, norm, bin_axis)

    dx = bins[1:] - bins[:-1]
    shape = [1] * n
    shape[bin_axis] = -1
    dx = reshape(xp, dx, shape)

    return y_pred / dx

def posterior_cdf(y_pred,
                  bins,
                  bin_axis=1):

    if len(y_pred.shape) == 1:
        bin_axis = 0
    n_y = y_pred.shape[bin_axis]
    n_b = len(bins)
    _check_dimensions(n_y, n_b)
    xp = get_array_module(y_pred)
    n = len(y_pred.shape)

    y_cdf = cumtrapz(xp, y_pred, bins, bin_axis)

    selection = [slice(0, None)] * n
    selection[bin_axis] = slice(-1, None)
    y_cdf = y_cdf / y_cdf[tuple(selection)]
    return y_cdf

def posterior_mean(y_pred,
                   bins,
                   bin_axis=1):
    if len(y_pred.shape) == 1:
        bin_axis = 0
    n_y = y_pred.shape[bin_axis]
    n_b = len(bins)
    _check_dimensions(n_y, n_b)
    xp = get_array_module(y_pred)
    n = len(y_pred.shape)

    shape = [1] * n
    shape[bin_axis] = -1
    bins_r = reshape(xp, 0.5 * (bins[1:] + bins[:-1]), shape)

    return trapz(xp, bins_r * y_pred, bins, bin_axis)

def probability_less_than(y_pred,
                          bins,
                          y,
                          bin_axis=1):
    if len(y_pred.shape) == 1:
        bin_axis = 0
    n_y = y_pred.shape[bin_axis]
    n_b = len(bins)
    _check_dimensions(n_y, n_b)
    xp = get_array_module(y_pred)
    n = len(y_pred.shape)

    x = 0.5 * (bins[1:] + bins[:-1])
    mask = x < y
    shape = [1] * n
    shape[bin_axis] = -1
    mask = as_type(xp, reshape(xp, mask , shape), y_pred)

    return trapz(xp, mask * y_pred, bins, bin_axis)


def posterior_quantiles(y_pred,
                        bins,
                        quantiles,
                        bin_axis=1):

    if len(y_pred.shape) == 1:
        bin_axis = 0
    n_y = y_pred.shape[bin_axis]
    n_b = len(bins)
    _check_dimensions(n_y, n_b)
    xp = get_array_module(y_pred)

    y_cdf = posterior_cdf(y_pred, bins, bin_axis=bin_axis)

    n = len(y_pred.shape)
    dx = bins[1:] - bins[:-1]
    x_shape = [1] * n
    x_shape[bin_axis] = numel(bins)
    dx = pad_zeros_left(xp, dx, 1, 0)
    dx = reshape(xp, dx, x_shape)

    y_qs = []
    for q in quantiles:
        mask = as_type(xp, y_cdf <= q, y_cdf)
        y_q = bins[0] + xp.sum(mask * dx, bin_axis)
        y_q = expand_dims(xp, y_q, bin_axis)
        y_qs.append(y_q)

    y_q = concatenate(xp, y_qs, bin_axis)
    return y_q

def posterior_median(y_pred,
                     bins,
                     bin_axis=1):
    quantiles = posterior_quantiles(y_pred, bins, [0.5], bin_axis=bin_axis)
    n = len(y_pred.shape)
    selection = [slice(0, None)] * n
    selection[bin_axis] = 0
    return quantiles[tuple(selection)]

def probability_larger_than(y_pred,
                            bins,
                            quantiles,
                            bin_axis=1):
    return 1.0 - probability_less_than(y_pred,
                                       bins,
                                       quantiles,
                                       bin_axis=bin_axis)
