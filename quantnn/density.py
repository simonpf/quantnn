"""
===============
quantnn.density
===============

This module provides generic functions to manipulate and derive statistics
from probabilistic predictions in the form of discretized probability density
functions as produced by DRNNs.
"""
from quantnn.common import InvalidDimensionException
from quantnn.generic import (
    get_array_module,
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
    sample_uniform,
    argmax,
    take_along_axis,
    digitize,
    scatter_add
)


def _check_dimensions(n_y, n_b):
    if n_y != n_b - 1:
        raise InvalidDimensionException(
            f"Dimensions of the provided array 'y_pred' ({n_y}) do not match the"
            f" provided number of bin edges ({n_b}). Note that there should be"
            f" one more bin edge than bin values in 'y_pred'."
        )


def normalize(y_pred, bins, bin_axis=1, density=False):
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

    dx = bins[1:] - bins[:-1]
    shape = [1] * n
    shape[bin_axis] = -1
    dx = reshape(xp, dx, shape)

    if density:
        norm = (y_pred * dx).sum(bin_axis)
        norm = expand_dims(xp, norm, bin_axis)
        return y_pred / norm

    norm = y_pred.sum(bin_axis)
    norm = expand_dims(xp, norm, bin_axis)
    return y_pred / norm / dx


def posterior_cdf(y_pdf, bins, bin_axis=1):
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


def posterior_mean(y_pdf, bins, bin_axis=1):
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


def posterior_std_dev(y_pdf, bins, bin_axis=1):
    """
    Calculate posterior standard deviation from predicted PDFs.

    Args:
        y_pdf: Tensor containing the predicted PDFs.
        bins: The bin-boundaries corresponding to the predictions.
        bin_axis: The index of the tensor axis which contains the predictions
            for each bin.

    Return:
        Tensor with rank reduced by one compared to ``y_pdf`` and with
        the values along ``bin_axis`` of ``y_pdf`` replaced with the
        std. dev. of the corresponding PDF.
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

    x = trapz(xp, bins_r * y_pdf, bins, bin_axis)
    x2 = trapz(xp, bins_r * bins_r * y_pdf, bins, bin_axis)
    return xp.sqrt(x2 - x ** 2)


def posterior_median(y_pred, bins, bin_axis=1):
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


def posterior_quantiles(y_pdf, bins, quantiles, bin_axis=1):
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
    n_dims = len(y_pdf.shape)

    _check_dimensions(n_y, n_b)
    xp = get_array_module(y_pdf)

    y_cdf = posterior_cdf(y_pdf, bins, bin_axis=bin_axis)

    n = len(y_pdf.shape)
    dx = bins[1:] - bins[:-1]
    x_shape = [1] * n
    x_shape[bin_axis] = numel(bins)
    dx = pad_zeros_left(xp, dx, 1, 0)
    dx = reshape(xp, dx, x_shape)

    selection = [slice(0, None)] * n_dims
    selection[bin_axis] = slice(0, -1)
    selection_l = tuple(selection)
    selection[bin_axis] = slice(1, None)
    selection_r = tuple(selection)
    cdf_l = y_cdf[selection_l]
    cdf_r = y_cdf[selection_r]
    d_cdf = cdf_r - cdf_l

    shape = [1] * n_dims
    shape[bin_axis] = -1
    bins_l = bins.reshape(shape)[selection_l]
    bins_r = bins.reshape(shape)[selection_r]

    y_qs = []
    for q in quantiles:
        mask_l = as_type(xp, cdf_l <= q, cdf_l)
        mask_r = as_type(xp, cdf_r > q, cdf_l)
        mask = mask_l * mask_r

        d_q = q - expand_dims(xp, (cdf_l * mask).sum(bin_axis), bin_axis)

        result = (d_q * bins_r + (d_cdf - d_q) * bins_l) * mask
        result = result / (d_cdf + as_type(xp, d_cdf < 1e-6, d_cdf))
        result = result.sum(bin_axis)
        result = result + bins[-1] * (1.0 - as_type(xp, mask_r.sum(bin_axis) > 0, mask))
        result = result + bins[0] * (1.0 - as_type(xp, mask_l.sum(bin_axis) > 0, mask))
        result = expand_dims(xp, result, bin_axis)

        y_qs.append(result)

    y_q = concatenate(xp, y_qs, bin_axis)
    return y_q


def probability_less_than(y_pdf, bins, y, bin_axis=1):
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
    mask = as_type(xp, reshape(xp, mask, shape), y_pdf)

    return trapz(xp, mask * y_pdf, bins, bin_axis)


def probability_larger_than(y_pred, bins, quantiles, bin_axis=1):
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
    return 1.0 - probability_less_than(y_pred, bins, quantiles, bin_axis=bin_axis)


def sample_posterior(y_pred, bins, n_samples=1, bin_axis=1):
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
        y_index[bin_axis] = slice(i, i + 1)
        y_r = y_cdf[tuple(y_index)]
        b_r = bins[i]

        mask = as_type(xp, (y_l < samples) * (y_r >= samples), y_l)
        results += b_l * (y_r - samples) * mask
        results += b_r * (samples - y_l) * mask
        results /= mask * (y_r - y_l) + (1.0 - mask)

        b_l = b_r
        y_l = y_r

    mask = as_type(xp, y_r < samples, y_r)
    results += mask * b_r
    return results


def crps(y_pdf, y_true, bins, bin_axis=1):
    r"""
    Compute the Continuous Ranked Probability Score (CRPS) for a given
    discrete probability density.

    This function uses a piece-wise linear fit to the approximate posterior
    CDF obtained from the predicted quantiles in :code:`y_pred` to
    approximate the continuous ranked probability score (CRPS):

    .. math::
        CRPS(\mathbf{y}, x) = \int_{-\infty}^\infty (F_{x | \mathbf{y}}(x')
        - \mathrm{1}_{x < x'})^2 \: dx'

    Args:

        y_pred: Tensor containing the predicted discrete posterior PDF
            with the probabilities for different bins oriented along axis
            ``bin_axis`` in ``y_pred``.

        y_true: Array containing the true point values.

        bins: 1D array containing the bins corresponding to the probabilities
            in ``y_pred``.

    Returns:

        Tensor of rank :math:`k - 1` containing the CRPS values for each of the
        predictions in ``y_pred``.
    """
    if len(y_pdf.shape) == 1:
        bin_axis = 0
    n_y = y_pdf.shape[bin_axis]
    n_b = len(bins)
    n_dims = len(y_pdf.shape)
    _check_dimensions(n_y, n_b)
    xp = get_array_module(y_pdf)
    n = len(y_pdf.shape)

    y_cdf = posterior_cdf(y_pdf, bins, bin_axis=bin_axis)

    x = bins
    shape = [1] * n_dims
    shape[bin_axis] = -1
    x = x.reshape(shape)

    if len(y_true.shape) < len(y_pdf.shape):
        y_true = y_true.unsqueeze(bin_axis)

    i = as_type(xp, x > y_true, y_cdf)
    crps = trapz(xp, (y_cdf - i) ** 2, x, bin_axis)
    return crps


def quantile_function(y_pdf, y_true, bins, bin_axis=1):
    """
    Evaluates the quantile function at given y values.
    """
    if len(y_pdf.shape) == 1:
        bin_axis = 0
    n_y = y_pdf.shape[bin_axis]
    n_b = len(bins)
    n_dims = len(y_pdf.shape)
    _check_dimensions(n_y, n_b)
    xp = get_array_module(y_pdf)
    n = len(y_pdf.shape)

    y_cdf = posterior_cdf(y_pdf, bins, bin_axis=bin_axis)
    selection = [slice(0, None)] * n_dims
    selection[bin_axis] = slice(0, -1)
    selection_l = tuple(selection)
    selection[bin_axis] = slice(1, None)
    selection_r = tuple(selection)
    cdf_l = y_cdf[selection_l]
    cdf_r = y_cdf[selection_r]

    shape = [1] * n_dims
    shape[bin_axis] = -1
    bins_l = bins.reshape(shape)[selection_l]
    bins_r = bins.reshape(shape)[selection_r]
    d_bins = bins_r - bins_l

    if len(y_true.shape) < len(y_pdf.shape):
        y_true = expand_dims(xp, y_true, bin_axis)

    mask_l = as_type(xp, bins_l <= y_true, y_true)
    mask_r = as_type(xp, bins_r > y_true, y_true)
    mask = mask_l * mask_r

    dy = y_true - expand_dims(xp, (bins_l * mask).sum(bin_axis), bin_axis)

    result = (dy * cdf_r + (d_bins - dy) * cdf_l) * mask
    result = result / d_bins
    result = result.sum(bin_axis)
    result = result + 1.0 - as_type(xp, mask_r.sum(bin_axis) > 0, mask)

    return result


def posterior_maximum(y_pdf, bins, bin_axis=1):
    """
    Calculate maximum of the posterior distribution.

    Args:
        y_pdf: Rank-``k`` Tensor containing the normalized, discrete
            posterior pdf.
        bins: Array containing the bin boundaries corresponding to 'y_pdf'.
        bin_axis: The axis along which the bins of the posterior PDF are
            oriented.

    Return:
        Tensor of rank ``k - 1`` containing the values corresponding to the
        maxima of the given posterior.
    """
    if len(y_pdf.shape) == 1:
        bin_axis = 0
    xp = get_array_module(y_pdf)
    x = 0.5 * (bins[1:] + bins[:-1])
    indices = argmax(xp, y_pdf, axes=bin_axis)
    return x[indices]


def add(y_pdf_1, bins_1, y_pdf_2, bins_2, bins_out, bin_axis=1):
    """
    Calculate the discretized PDF of the sum of two random variables
    represented by their respective discretized PDFs.

    Args:
        y_pdf_1: The discretized PDF of the first random variable.
        bins_1: The bin boundaries corresponding to 'y_pdf_1'.
        y_pdf_2: The discretized PDF of the second random variable.
        bins_2: The bin boundaries corresponding to 'y_pdf_2'.
        bins_out: The bins boundaries for the resulting discretized PDF.
        bin_axis: The dimension along which the probabilities are
            oriented.

    Return:

        A tensor containing the discretized PDF corresponding to the sum of
        the two given PDFs.
    """
    if len(y_pdf_1.shape) == 1:
        bin_axis = 0
    xp = get_array_module(y_pdf_1)

    bins_1_c = 0.5 * (bins_1[1:] + bins_1[:-1])
    dx_1 = bins_1[1:] - bins_1[:-1]
    shape_1 = [1] * len(y_pdf_1.shape)
    shape_1[bin_axis] = numel(bins_1) - 1
    dx_1 = dx_1.reshape(shape_1)
    p_1 = y_pdf_1 * dx_1

    bins_2_c = 0.5 * (bins_2[1:] + bins_2[:-1])
    dx_2 = bins_2[1:] - bins_2[:-1]
    shape_2 = [1] * len(y_pdf_2.shape)
    shape_2[bin_axis] = numel(bins_2) - 1
    dx_2 = dx_2.reshape(shape_2)
    p_2 = y_pdf_2 * dx_2

    out_shape = list(y_pdf_1.shape)
    out_shape[bin_axis] = numel(bins_out) - 1
    p_out = zeros(xp, out_shape, like=y_pdf_1)

    rank = len(y_pdf_1.shape)
    selection = [slice(0, None)] * rank

    n_bins = numel(bins_1_c)
    offsets = sample_uniform(xp, (n_bins,), like=bins_2)
    for i in range(n_bins):
        d_b = bins_1[i + 1] - bins_1[i]
        b = bins_1[i] + offsets[i] * d_b
        selection[bin_axis] = i
        bins = bins_2_c + b
        probs = p_1[tuple(selection)] * p_2
        inds = digitize(xp, bins, bins_out) - 1
        p_out = scatter_add(xp, p_out, inds, probs, bin_axis)

    return normalize(p_out, bins_out, bin_axis=bin_axis)
