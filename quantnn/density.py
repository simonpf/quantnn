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
                             as_type,
                             zeros,
                             sample_uniform)

def _check_dimensions(n_y, n_b):
    if n_y != n_b - 1:
        raise InvalidDimensionException(
            f"Dimensions of the provided array 'y_pred' ({n_y}) do not match the"
            f" provided number of bin edges ({n_b}). Note that there should be"
            f" one more bin edge than bin values in 'y_pred'."
        )

def normalize(y_pred,
              bins,
              bin_axis=1):
    """
    Converts the raw DRNN output to a PDF.

    Args:
        y_pred: Tensor of predictions from a DRNN with the bin-probabilities
            along one of its axes.
        bins: The bin boundaries corresponding to the predictions.
        bin_axis: The index of the tensor axis which contains the predictions
            for each bin.

    Return:
        Tensor with the same shape as ``y_pred`` but with the values
        transformed to represent the PDF corresponding to the predicted
        bin probabilities in ``y_pred``.
    """
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

def posterior_cdf(y_pdf,
                  bins,
                  bin_axis=1):
    """
    Calculate CDF from predicted probability density function.

    Args:
        y_pdf: Tensor containing the predicted PDFs.
        bins: The bin-boundaries corresponding to the predictions.
        bin_axis: The index of the tensor axis which contains the predictions
            for each bin.

    Return:
        Tensor with the same shape as ``y_pdf`` but with the values
        transformed to represent the CDF corresponding to the predicted
        PDF in ``y_pdf``.
    """
    if len(y_pdf.shape) == 1:
        bin_axis = 0
    n_y = y_pdf.shape[bin_axis]
    n_b = len(bins)
    _check_dimensions(n_y, n_b)
    xp = get_array_module(y_pdf)
    n = len(y_pdf.shape)

    y_cdf = cumtrapz(xp, y_pdf, bins, bin_axis)

    selection = [slice(0, None)] * n
    selection[bin_axis] = slice(-1, None)
    y_cdf = y_cdf / y_cdf[tuple(selection)]
    return y_cdf

def posterior_mean(y_pdf,
                   bins,
                   bin_axis=1):
    """
    Calculate posterior mean from predicted PDFs.

    Args:
        y_pdf: Tensor containing the predicted PDFs.
        bins: The bin-boundaries corresponding to the predictions.
        bin_axis: The index of the tensor axis which contains the predictions
            for each bin.

    Return:
        Tensor with rank reduced by one compared to ``y_pdf`` and with
        the values along ``bin_axis`` of ``y_pdf`` replaced with the
        mean value of the PDF.
    """
    if len(y_pdf.shape) == 1:
        bin_axis = 0
    n_y = y_pdf.shape[bin_axis]
    n_b = len(bins)
    _check_dimensions(n_y, n_b)
    xp = get_array_module(y_pdf)
    n = len(y_pdf.shape)

    shape = [1] * n
    shape[bin_axis] = -1
    bins_r = reshape(xp, 0.5 * (bins[1:] + bins[:-1]), shape)

    return trapz(xp, bins_r * y_pdf, bins, bin_axis)


def posterior_median(y_pred,
                     bins,
                     bin_axis=1):
    """
    Calculate the posterior median from predicted PDFs.

    Args:
        y_pdf: Tensor containing the predicted PDFs.
        bins: The bin-boundaries corresponding to the predictions.
        bin_axis: The index of the tensor axis which contains the predictions
            for each bin.

    Return:
        Tensor with rank reduced by one compared to ``y_pdf`` and with
        the values along ``bin_axis`` of ``y_pdf`` replaced with the
        median value of the PDFs.
    """
    quantiles = posterior_quantiles(y_pred, bins, [0.5], bin_axis=bin_axis)
    n = len(y_pred.shape)
    selection = [slice(0, None)] * n
    selection[bin_axis] = 0
    return quantiles[tuple(selection)]


def posterior_quantiles(y_pdf,
                        bins,
                        quantiles,
                        bin_axis=1):
    """
    Calculate posterior quantiles from predicted PDFs.

    Args:
        y_pdf: Tensor containing the predicted PDFs.
        bins: The bin-boundaries corresponding to the predictions.
        quantiles: List containing the quantiles fractions of the quantiles
             to compute.
        bin_axis: The index of the tensor axis which contains the predictions
            for each bin.

    Return:
        Tensor with same rank as ``y_pdf`` but with the values
        the values along ``bin_axis`` replaced with the quantiles
        of the predicted distributions.
    """
    if len(y_pdf.shape) == 1:
        bin_axis = 0
    n_y = y_pdf.shape[bin_axis]
    n_b = len(bins)
    _check_dimensions(n_y, n_b)
    xp = get_array_module(y_pdf)

    y_cdf = posterior_cdf(y_pdf, bins, bin_axis=bin_axis)

    n = len(y_pdf.shape)
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

def probability_less_than(y_pdf,
                          bins,
                          y,
                          bin_axis=1):
    """
    Calculate the probability of a sample being less than a given
    value for a tensor of predicted PDFs.

    Args:
        y_pdf: Tensor containing the predicted PDFs.
        bins: The bin-boundaries corresponding to the predictions.
        y: The sample value.
        bin_axis: The index of the tensor axis which contains the predictions
            for each bin.

    Return:
        Tensor with rank reduced by one compared to ``y_pdf`` and with
        the values along ``bin_axis`` of ``y_pdf`` replaced with the
        probability that a sample of the distribution is smaller than
        the given value ``y``.
    """
    if len(y_pdf.shape) == 1:
        bin_axis = 0
    n_y = y_pdf.shape[bin_axis]
    n_b = len(bins)
    _check_dimensions(n_y, n_b)
    xp = get_array_module(y_pdf)
    n = len(y_pdf.shape)

    x = 0.5 * (bins[1:] + bins[:-1])
    mask = x < y
    shape = [1] * n
    shape[bin_axis] = -1
    mask = as_type(xp, reshape(xp, mask , shape), y_pdf)

    return trapz(xp, mask * y_pdf, bins, bin_axis)


def probability_larger_than(y_pred,
                            bins,
                            quantiles,
                            bin_axis=1):
    """
    Calculate the probability of a sample being larger than a given
    value for a tensor of predicted PDFs.

    Args:
        y_pdf: Tensor containing the predicted PDFs.
        bins: The bin-boundaries corresponding to the predictions.
        y: The sample value.
        bin_axis: The index of the tensor axis which contains the predictions
            for each bin.

    Return:
        Tensor with rank reduced by one compared to ``y_pdf`` and with
        the values along ``bin_axis`` of ``y_pdf`` replaced with the
        probability that a sample of the distribution is larger than
        the given value ``y``.
    """
    return 1.0 - probability_less_than(y_pred,
                                       bins,
                                       quantiles,
                                       bin_axis=bin_axis)

def sample_posterior(y_pred,
                     bins,
                     n_samples=1,
                     bin_axis=1):
    """
    Sample the posterior distribution described by the predicted PDF.

    The sampling is performed by interpolating the inverse of the cumulative
    distribution function to value sampled from a uniform distribution.

    Args:
        y_pred: A rank-k tensor containing the predicted bin-probabilities
            along the axis specified by ``quantile_axis``.
        bins: The bin bounrdaries corresponding to the predicted
            bin probabilities.
        n_samples: How many samples to generate for each prediction.
        bin_axis: The axis in y_pred along which the predicted bin
             probabilities are located.

    Returns:
        A rank-k tensor with the values along ``bin_axis`` replaced by
        samples of the posterior distribution.
    """
    if len(y_pred.shape) == 1:
        bin_axis = 0
    xp = get_array_module(y_pred)
    n_dims = len(y_pred.shape)
    y_cdf = posterior_cdf(y_pred, bins, bin_axis=bin_axis)

    n_bins = len(bins)


    output_shape = list(y_cdf.shape)
    output_shape[bin_axis] = n_samples
    results = zeros(xp, output_shape, like=y_pred)

    y_index = [slice(0, None)] * n_dims
    y_index[bin_axis] = slice(0, 1) 
    y_l = y_cdf[tuple(y_index)]
    b_l = bins[0]

    samples = as_type(xp, sample_uniform(xp, tuple(output_shape)), y_cdf)

    for i in range(1, n_bins):
        y_index = [slice(0, None)] * n_dims
        y_index[bin_axis] = slice(i, i+1)
        y_r = y_cdf[tuple(y_index)]
        b_r = bins[i]

        mask = as_type(xp, (y_l < samples) * (y_r >= samples), y_l)
        results += b_l * (y_r - samples) * mask
        results += b_r * (samples - y_l) * mask
        results /= (mask * (y_r - y_l) + (1.0 - mask))

        b_l = b_r
        y_l = y_r


    mask = as_type(xp, y_r < samples, y_r)
    results += mask * b_r
    return results
