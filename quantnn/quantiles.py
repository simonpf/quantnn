"""
quantnn.functional
==================

The ``quantnn.functional`` module provides functions to calculate relevant
 statistics from QRNN output.
"""
from copy import copy

import numpy as np
from scipy.stats import norm

from quantnn.common import InvalidDimensionException
from quantnn.generic import (
    arange,
    get_array_module,
    to_array,
    sample_uniform,
    sample_gaussian,
    numel,
    expand_dims,
    concatenate,
    pad_zeros,
    as_type,
    cumtrapz,
    trapz,
    reshape,
    zeros,
    ones,
    cumsum,
    argmax,
    take_along_axis
)


def cdf(y_pred, quantiles, quantile_axis=1):
    """
    Calculates the cumulative distribution function (CDF) from predicted
    quantiles.

    This method  extends the quantiles in 'y_pred' to  0 and 1 by
    extending the first and last segments with a 50% reduction in the
    slope.

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
    xp = get_array_module(y_pred)

    y_cdf = quantiles

    y_cdf = concatenate(
        xp, [zeros(xp, 1, like=y_cdf), y_cdf, ones(xp, 1, like=y_cdf)], 0
    )

    selection = [slice(0, None)] * len(y_pred.shape)
    selection_c = copy(selection)
    selection_c[quantile_axis] = 0
    selection_c = tuple(selection_c)
    selection_r = copy(selection)
    selection_r[quantile_axis] = 1
    selection_r = tuple(selection_r)
    dx = y_pred[selection_r] - y_pred[selection_c]
    dx /= quantiles[1] - quantiles[0]
    x_cdf_l = y_pred[selection_c] - 2.0 * quantiles[0] * dx
    x_cdf_l = expand_dims(xp, x_cdf_l, quantile_axis)

    selection_l = copy(selection)
    selection_l[quantile_axis] = -2
    selection_l = tuple(selection_l)
    selection_c = copy(selection)
    selection_c[quantile_axis] = -1
    selection_c = tuple(selection_c)
    dx = y_pred[selection_c] - y_pred[selection_l]
    dx /= quantiles[-1] - quantiles[-2]
    x_cdf_r = y_pred[selection_c] + 2.0 * (1.0 - quantiles[-1]) * dx
    x_cdf_r = expand_dims(xp, x_cdf_r, quantile_axis)

    x_cdf = concatenate(xp, [x_cdf_l, y_pred, x_cdf_r], quantile_axis)

    return x_cdf, y_cdf


def pdf(y_pred, quantiles, quantile_axis=1):
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

    xp = get_array_module(y_pred)

    x_cdf, y_cdf = cdf(y_pred, quantiles, quantile_axis=quantile_axis)

    output_shape = list(x_cdf.shape)
    output_shape[quantile_axis] += 1
    n_dims = len(output_shape)

    #
    # Assemble x-tensor
    #

    selection_l = [slice(0, None)] * n_dims
    selection_l[quantile_axis] = slice(0, -1)
    selection_l = tuple(selection_l)
    selection_r = [slice(0, None)] * n_dims
    selection_r[quantile_axis] = slice(1, None)
    selection_r = tuple(selection_r)
    x_pdf = 0.5 * (x_cdf[selection_l] + x_cdf[selection_r])

    selection = [slice(0, None)] * n_dims
    selection[quantile_axis] = 0
    x_pdf_l = x_cdf[tuple(selection)]
    x_pdf_l = expand_dims(xp, x_pdf_l, quantile_axis)

    selection[quantile_axis] = -1
    x_pdf_r = x_cdf[tuple(selection)]
    x_pdf_r = expand_dims(xp, x_pdf_r, quantile_axis)

    x_pdf = concatenate(xp, [x_pdf_l, x_pdf, x_pdf_r], quantile_axis)

    #
    # Assemble y-tensor
    #

    shape = [1] * n_dims
    shape[quantile_axis] = -1
    y_pdf = 1.0 / (x_cdf[selection_r] - x_cdf[selection_l])
    y_pdf = y_pdf * (y_cdf[1:] - y_cdf[:-1]).reshape(shape)
    y_pdf = pad_zeros(xp, y_pdf, 1, quantile_axis)

    return x_pdf, y_pdf


def pdf_binned(y_pred, quantiles, bins, quantile_axis=1):
    """
    Calculate binned representation of the posterior probability density
    function (PDF).

    The binned PDF is simple calculated by linearly interpolating the
    piece-wise linear PDF computed using the :py:meth`pdf` method.

    Args:
        y_pred: Rank-k Tensor containing the predicted quantiles along the
            quantile axis.
        quantiles: The quantile fractions corresponding to the predicted
            quantiles in y_pred.
        bins: Rank-1 tensor containing the ``n_bins`` boundaries for the bins
            to use to bin the PDF.
        quantile_axis: The axis of y_pred along which the predicted
            quantiles are located.

    Returns:
        Rank-k tensor with ``n_bins - 1`` elements along ``quantile_axis``
        containing the probability of the result to fall between the
        corresponding bin edges.
    """
    if len(y_pred.shape) == 1:
        quantile_axis = 0

    xp = get_array_module(y_pred)
    n = len(y_pred.shape)
    x_cdf, y_cdf = cdf(y_pred, quantiles, quantile_axis=quantile_axis)

    y_cdf_shape = [1] * n
    y_cdf_shape[quantile_axis] = -1
    y_cdf = reshape(xp, y_cdf, y_cdf_shape)

    selection_l = [slice(0, None)] * n
    selection_l[quantile_axis] = slice(0, -1)
    selection_l = tuple(selection_l)
    selection_r = [slice(0, None)] * n
    selection_r[quantile_axis] = slice(1, None)
    selection_r = tuple(selection_r)

    selection_le = [slice(0, None)] * n
    selection_le[quantile_axis] = 0
    selection_le = tuple(selection_le)

    selection_re = [slice(0, None)] * n
    selection_re[quantile_axis] = -1
    selection_re = tuple(selection_re)

    y_pdf_binned = []

    #
    # Interpolate CDF for leftmost bin boundary.
    #
    b_l = bins[0]
    mask_r = as_type(xp, (x_cdf[selection_r] >= b_l), y_cdf)
    mask_l = as_type(xp, (x_cdf[selection_l] < b_l), y_cdf)
    mask = mask_l * mask_r

    mask_xr = as_type(xp, xp.sum(mask_r, quantile_axis) == 0.0, mask_r)
    mask_xl = as_type(xp, xp.sum(mask_l, quantile_axis) == 0.0, mask_l)

    x_cdf_l = xp.sum(x_cdf[selection_l] * mask, quantile_axis)
    x_cdf_r = xp.sum(x_cdf[selection_r] * mask, quantile_axis)
    d = (x_cdf_r - x_cdf_l) + (1.0 - xp.sum(mask, quantile_axis))
    w_cdf_l = (x_cdf_r - b_l) / d
    w_cdf_r = (b_l - x_cdf_l) / d

    y_cdf_l = (
        xp.sum(mask * y_cdf[selection_l] * mask, quantile_axis) * w_cdf_l
        + xp.sum(mask * y_cdf[selection_r] * mask, quantile_axis) * w_cdf_r
        + mask_xl * y_cdf[selection_le]
        + mask_xr * y_cdf[selection_re]
    )

    for i in range(len(bins) - 1):

        b_r = bins[i + 1]

        #
        # Interpolate CDF for right bin boundary.
        #
        mask_r = as_type(xp, (x_cdf[selection_r] >= b_r), y_cdf)
        mask_l = as_type(xp, (x_cdf[selection_l] < b_r), y_cdf)
        mask = mask_l * mask_r

        mask_xr = as_type(xp, xp.sum(mask_r, quantile_axis) == 0.0, mask_r)
        mask_xl = as_type(xp, xp.sum(mask_l, quantile_axis) == 0.0, mask_l)

        x_cdf_l = xp.sum(x_cdf[selection_l] * mask, quantile_axis)
        x_cdf_r = xp.sum(x_cdf[selection_r] * mask, quantile_axis)
        d = (x_cdf_r - x_cdf_l) + (1.0 - xp.sum(mask, quantile_axis))
        w_cdf_l = (x_cdf_r - b_r) / d
        w_cdf_r = (b_r - x_cdf_l) / d

        y_cdf_r = (
            xp.sum(mask * y_cdf[selection_l] * mask, quantile_axis) * w_cdf_l
            + xp.sum(mask * y_cdf[selection_r] * mask, quantile_axis) * w_cdf_r
            + mask_xl * y_cdf[selection_le]
            + mask_xr * y_cdf[selection_re]
        )

        dy_cdf = expand_dims(xp, y_cdf_r - y_cdf_l, quantile_axis)
        y_pdf_binned.append(dy_cdf / (b_r - b_l))
        y_cdf_l = y_cdf_r
        b_l = b_r

    return concatenate(xp, y_pdf_binned, quantile_axis)


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
    xp = get_array_module(y_pred)

    x_cdf, y_cdf = cdf(y_pred, quantiles, quantile_axis=quantile_axis)
    return trapz(xp, x_cdf, y_cdf, quantile_axis)


def posterior_std_dev(y_pred, quantiles, quantile_axis=1):
    r"""
    Computes the standard deviation of the posterior distribution defined by
    an array of predicted quantiles.

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
    xp = get_array_module(y_pred)

    x_cdf, y_cdf = cdf(y_pred, quantiles, quantile_axis=quantile_axis)
    x_mean =  trapz(xp, x_cdf, y_cdf, quantile_axis)
    x2_mean =  trapz(xp, x_cdf * x_cdf, y_cdf, quantile_axis)
    return xp.sqrt(x2_mean - x_mean ** 2)


def posterior_median(y_pred, quantiles, quantile_axis=1):
    r"""
    Computes the median of the posterior distribution defined by an array
    of predicted quantiles.

    Args:
        y_pred: A rank-k tensor of predicted quantiles with the quantiles
             located along the axis given by ``quantile_axis``.
        quantiles: The quantile fractions corresponding to the quantiles
             located along the quantile axis.
        quantile_axis: The axis along which the quantiles are located.

    Returns:

        Rank k-1 tensor containing the posterior median for the provided inputs.
    """
    if len(y_pred.shape) == 1:
        quantile_axis = 0
    xp = get_array_module(y_pred)

    n = len(y_pred.shape)
    indices = arange(xp, 0, len(quantiles), 1.0)
    mask = (quantiles[1:] > 0.5) * (quantiles[:-1] <= 0.5)

    selection = [slice(0, None)] * n

    index = indices[:-1][mask]
    if len(index) == 0:
        if quantiles[0] < 0.5:
            selection[quantile_axis] = 0
            selection_l = tuple(selection)
            return y_pred[selection_l]
        else:
            selection[quantile_axis] = -1
            selection_r = tuple(selection)
            return y_pred[selection_r]

    index = int(index[0])
    d = quantiles[index + 1] - quantiles[index]
    w_l = (quantiles[index + 1] - 0.5) / d
    w_r = (0.5 - quantiles[index]) / d

    selection = [slice(0, None)] * n
    selection[quantile_axis] = index
    selection_l = tuple(selection)
    selection[quantile_axis] = index + 1
    selection_r = tuple(selection)

    return w_l * y_pred[selection_l] + w_r * y_pred[selection_r]


def posterior_quantiles(y_pred, quantiles, new_quantiles, quantile_axis=1):
    r"""
    Computes the median of the posterior distribution defined by an array
    of predicted quantiles.

    Args:
        y_pred: A rank-k tensor of predicted quantiles with the quantiles
             located along the axis given by ``quantile_axis``.
        quantiles: The quantile fractions corresponding to the quantiles
             located along the quantile axis.
        quantile_axis: The axis along which the quantiles are located.

    Returns:

        Rank k-1 tensor containing the posterior median for the provided inputs.
    """
    if len(y_pred.shape) == 1:
        quantile_axis = 0
    xp = get_array_module(y_pred)

    n = len(y_pred.shape)
    indices = as_type(xp, arange(xp, 0, len(quantiles), 1.0), y_pred)
    selection = [slice(0, None)] * n

    y_qs = []

    for q in new_quantiles:
        mask_l = quantiles <= q
        mask_r = quantiles > q

        index_l = indices[mask_l]
        if len(index_l) == 0:
            selection[quantile_axis] = 0
            selection_l = tuple(selection)
            y_q = expand_dims(xp, y_pred[selection_l], quantile_axis)
            y_qs.append(y_q)
            continue

        index_r = indices[mask_r]
        if len(index_r) == 0:
            selection[quantile_axis] = -1
            selection_r = tuple(selection)
            y_q = expand_dims(xp, y_pred[selection_r], quantile_axis)
            y_qs.append(y_q)
            continue

        index = int(index_l[-1])

        d = quantiles[index + 1] - quantiles[index]
        w_l = (quantiles[index + 1] - q) / d
        w_r = (q - quantiles[index]) / d

        selection = [slice(0, None)] * n
        selection[quantile_axis] = index
        selection_l = tuple(selection)
        selection[quantile_axis] = index + 1
        selection_r = tuple(selection)

        y_q = w_l * y_pred[selection_l] + w_r * y_pred[selection_r]
        y_q = expand_dims(xp, y_q, quantile_axis)
        y_qs.append(y_q)

    return concatenate(xp, y_qs, quantile_axis)


def crps(y_pred, y_true, quantiles, quantile_axis=1):
    r"""
    Compute the Continuous Ranked Probability Score (CRPS) for given
    predicted quantiles.

    This function uses a piece-wise linear fit to the approximate posterior
    CDF obtained from the predicted quantiles in :code:`y_pred` to
    approximate the continuous ranked probability score (CRPS):

    .. math::
        CRPS(\mathbf{y}, x) = \int_{-\infty}^\infty (F_{x | \mathbf{y}}(x')
        - \mathrm{1}_{x < x'})^2 \: dx'

    Args:

        y_pred: Tensor containing the predicted quantiles along the axis
                specified by ``quantile_axis``.

        y_true: Array containing the true point values.

        quantiles: 1D array containing the quantile fractions corresponding
            corresponding to the predicted quantiles.


    Returns:

        Tensor of rank :math:`k - 1` containing the CRPS values for each of the
        predictions in ``y_pred``.
    """
    if len(y_pred.shape) == 1:
        quantile_axis = 0
    xp = get_array_module(y_pred)
    n_dims = len(y_pred.shape)

    x_cdf, y_cdf = cdf(y_pred, quantiles, quantile_axis=quantile_axis)

    y_true_shape = list(x_cdf.shape)
    y_true_shape[quantile_axis] = 1
    y_true = to_array(xp, y_true)
    y_true = reshape(xp, y_true, y_true_shape)

    mask = as_type(xp, x_cdf > y_true, y_pred)
    ind = ones(xp, x_cdf.shape, like=y_pred) * mask

    output_shape = list(x_cdf.shape)
    del output_shape[quantile_axis]
    integral = zeros(xp, output_shape, like=y_pred)
    x_index = [slice(0, None)] * n_dims

    y_l = y_cdf[0]
    x_index[quantile_axis] = 0
    x_l = x_cdf[tuple(x_index)]
    ind_l = ind[tuple(x_index)]

    for i in range(1, len(y_cdf)):

        y_r = y_cdf[i]
        x_index[quantile_axis] = i
        x_r = x_cdf[tuple(x_index)]
        ind_r = ind[tuple(x_index)]

        result = (ind_l - y_l) ** 2
        result += (ind_r - y_r) ** 2
        dx = x_r - x_l
        result *= 0.5 * dx
        integral += result

        y_l = y_r
        x_l = x_r
        ind_l = ind_r

    return integral


def probability_less_than(y_pred, quantiles, y, quantile_axis=1):
    """
    Calculate the probability that the predicted value is less
    than a given threshold value ``y`` given a tensor of predicted
    quantiles ``y_pred``.

    The probability :math:`P(Y > y)` is calculated by using the predicted
    quantiles to estimate the CDF of the posterior distribution, which
    is then interpolate to the given threshold value.

    Args:
        y_pred: A rank-k tensor containing the predicted quantiles along the
            axis specified by ``quantile_axis``.
        quantiles: The quantile fractions corresponding to the predicted
            quantiles.
        y: The threshold value.
        quantile_axis: The axis in y_pred along which the predicted quantiles
             are found.

    Returns:
         A rank-(k-1) tensor containing for each set of predicted quantiles the
         estimated probability of the true value being larger than the given
         threshold.
    """
    if len(y_pred.shape) == 1:
        quantile_axis = 0
    xp = get_array_module(y_pred)
    n_dims = len(y_pred.shape)
    x_cdf, y_cdf = cdf(y_pred, quantiles, quantile_axis=quantile_axis)

    output_shape = list(x_cdf.shape)
    del output_shape[quantile_axis]
    probabilities = zeros(xp, output_shape, like=y_pred)

    y_l = y_cdf[0]
    x_index = [slice(0, None)] * n_dims
    x_index[quantile_axis] = 0
    x_l = x_cdf[tuple(x_index)]

    for i in range(1, len(y_cdf)):
        y_r = y_cdf[i]
        x_index[quantile_axis] = i
        x_r = x_cdf[tuple(x_index)]

        mask = as_type(xp, (x_l < y) * (x_r >= y), x_l)
        probabilities += y_l * (x_r - y) * mask
        probabilities += y_r * (y - x_l) * mask
        probabilities /= mask * (x_r - x_l) + (1.0 - mask)

        y_l = y_r
        x_l = x_r

    mask = as_type(xp, x_r < y, x_r)
    probabilities += mask
    return probabilities


def probability_larger_than(y_pred, quantiles, y, quantile_axis=1):
    """
    Calculate the probability that the predicted value is larger
    than a given threshold value ``y`` given a tensor of predicted
    quantiles ``y_pred``.

    This simply calculates the complement of ``probability_less_than``.

    Args:
        y_pred: A rank-k tensor containing the predicted quantiles along the
            axis specified by ``quantile_axis``.
        quantiles: The quantile fractions corresponding to the predicted
            quantiles.
        y: The threshold value.
        quantile_axis: The axis in y_pred along which the predicted quantiles
             are found.

    Returns:
         A rank-(k-1) tensor containing for each set of predicted quantiles the
         estimated probability of the true value being larger than the given
         threshold.
    """
    return 1.0 - probability_less_than(
        y_pred, quantiles, y, quantile_axis=quantile_axis
    )


def sample_posterior(y_pred, quantiles, n_samples=1, quantile_axis=1):
    """
    Sample the posterior distribution described by the predicted quantiles.

    The sampling is performed by interpolating the inverse of the cumulative
    distribution function to value sampled from a uniform distribution.

    Args:
        y_pred: A rank-k tensor containing the predicted quantiles along the
            axis specified by ``quantile_axis``.
        quantiles: The quantile fractions corresponding to the predicted
            quantiles.
        n_samples: How many samples to generate for each prediction.
        quantile_axis: The axis in y_pred along which the predicted quantiles
             are found.

    Returns:
        A rank-k tensor with the values along ``quantile_axis`` replaced by
        samples of the posterior distribution.
    """
    if len(y_pred.shape) == 1:
        quantile_axis = 0
    xp = get_array_module(y_pred)
    n_dims = len(y_pred.shape)
    x_cdf, y_cdf = cdf(y_pred, quantiles, quantile_axis=quantile_axis)

    output_shape = list(y_pred.shape)
    output_shape[quantile_axis] = n_samples

    samples = as_type(xp, sample_uniform(xp, tuple(output_shape)), y_cdf)
    results = zeros(xp, samples.shape, like=y_pred)

    y_l = y_cdf[0]
    x_index = [slice(0, None)] * n_dims
    x_index[quantile_axis] = slice(0, 1)
    x_l = x_cdf[tuple(x_index)]

    for i in range(1, len(y_cdf)):
        y_r = y_cdf[i]
        x_index[quantile_axis] = slice(i, i + 1)
        x_r = x_cdf[tuple(x_index)]

        mask = as_type(xp, (samples > y_l) * (samples <= y_r), y_l)
        results += (x_l * (y_r - samples)) * mask
        results += (x_r * (samples - y_l)) * mask
        results /= mask * (y_r - y_l) + (1.0 - mask)

        y_l = y_r
        x_l = x_r

    return results


def fit_gaussian_to_quantiles(y_pred, quantiles, quantile_axis=1):
    """
    Fits Gaussian distributions to predicted quantiles.

    Fits mean and standard deviation values to quantiles by minimizing
    the mean squared distance of the predicted quantiles and those of
    the corresponding Gaussian distribution.

    Args:
        y_pred: A rank-k tensor containing the predicted quantiles along
            the axis specified by ``quantile_axis``.
        quantiles: Array of shape `(m,)` containing the quantile
            fractions corresponding to the predictions in ``y_pred``.

    Returns:
        Tuple ``(mu, sigma)`` of tensors of rank k-1 containing the mean and
        standard deviations of the Gaussian distributions corresponding to
        the predictions in ``y_pred``.
    """
    if len(y_pred.shape) == 1:
        quantile_axis = 0
    xp = get_array_module(y_pred)
    x = to_array(xp, norm.ppf(quantiles))
    n_dims = len(y_pred.shape)
    x_shape = [
        1,
    ] * n_dims
    x_shape[quantile_axis] = -1
    x_shape = tuple(x_shape)
    x = reshape(xp, x, x_shape)

    output_shape = list(y_pred.shape)
    output_shape[quantile_axis] = 1
    output_shape = tuple(output_shape)

    d2e_00 = numel(x)
    d2e_01 = x.sum()
    d2e_10 = x.sum()
    d2e_11 = (x ** 2).sum()

    d2e_det_inv = 1.0 / (d2e_00 * d2e_11 - d2e_01 * d2e_11)
    d2e_inv_00 = d2e_det_inv * d2e_11
    d2e_inv_01 = -d2e_det_inv * d2e_01
    d2e_inv_10 = -d2e_det_inv * d2e_10
    d2e_inv_11 = d2e_det_inv * d2e_00

    x = as_type(xp, reshape(xp, x, x_shape), y_pred)
    de_0 = reshape(xp, -(y_pred - x).sum(axis=quantile_axis), output_shape)
    de_1 = reshape(xp, -(x * (y_pred - x)).sum(axis=quantile_axis), output_shape)

    mu = -(d2e_inv_00 * de_0 + d2e_inv_01 * de_1)
    sigma = 1.0 - (d2e_inv_10 * de_0 + d2e_inv_11 * de_1)

    return mu, sigma


def sample_posterior_gaussian(y_pred, quantiles, n_samples=1, quantile_axis=1):
    """
    Sample the posterior distribution described by the predicted quantiles.

    The sampling is performed by fitting a Gaussian to the predicted a
    posteriori distribution and sampling from it.

    Args:
        y_pred: A rank-k tensor containing the predicted quantiles along the
            axis specified by ``quantile_axis``.
        quantiles: The quantile fractions corresponding to the predicted
            quantiles.
        n_samples: How many samples to generate for each prediction.
        quantile_axis: The axis in y_pred along which the predicted quantiles
             are found.

    Returns:
        A rank-k tensor with the values along ``quantile_axis`` replaced by
        samples of the posterior distribution.
    """
    if len(y_pred.shape) == 1:
        quantile_axis = 0
    xp = get_array_module(y_pred)
    mu, sigma = fit_gaussian_to_quantiles(
        y_pred, quantiles, quantile_axis=quantile_axis
    )

    output_shape = list(y_pred.shape)
    output_shape[quantile_axis] = n_samples
    samples = sample_gaussian(xp, tuple(output_shape))
    return mu + sigma * samples


def quantile_loss(y_pred, quantiles, y_true, quantile_axis=1):
    """
    Calculate the quantile loss for all predicted quantiles.

    Args:
        y_pred: A k-tensor containing the predicted quantiles along the
             axis specified by ``quantile_axis``.
        y_true: A tensor of rank k-1 containing the corresponding true
             values.
        quantiles: A vector or list containing the quantile fractions
             corresponding to the predicted quantiles.
        quantile_axis: The axis along which ``y_pred`` contains the
             the predicted quantiles.
    """
    if len(y_pred.shape) == 1:
        quantile_axis = 0
    xp = get_array_module(y_pred)
    n_dims = len(y_pred.shape)

    y_true_shape = list(y_pred.shape)
    y_true_shape[quantile_axis] = 1
    try:
        y_true = reshape(xp, y_true, y_true_shape)
    except Exception:
        raise InvalidDimensionException(
            "Could not reshape 'y_true' argument into expected shape "
            f"{y_true_shape}."
        )

    quantiles = to_array(xp, quantiles)
    quantiles_shape = [1] * n_dims
    quantiles_shape[quantile_axis] = len(quantiles)
    quantiles = reshape(xp, quantiles, quantiles_shape)

    dy = y_pred - y_true
    loss = zeros(xp, dy.shape, like=y_pred)
    mask = as_type(xp, dy > 0.0, dy)
    loss += mask * ((1.0 - quantiles) * dy)
    loss += -(1.0 - mask) * (quantiles * dy)
    return loss


def correct_a_priori(y_pred, quantiles, r, quantile_axis=1):
    """
    Correct predicted quantiles for a priori.

    Args:
        y_pred: Rank-k tensor containing the predicted quantiles along
            the axis given by 'quantile_axis'.
        quantiles: Rank-1 tensor containing the quantile fractions that
            correspond to the predicted quantiles.
        r: A priori density ratio to use to correct the observations.
        quantile_axis: The axis along which the quantile are oriented
            in 'y_pred'.
    """
    if len(y_pred.shape) == 1:
        quantile_axis = 0
    xp = get_array_module(y_pred)
    n_dims = len(y_pred.shape)
    x_pdf, y_pdf = pdf(y_pred, quantiles, quantile_axis=quantile_axis)

    selection = [slice(0, None)] * len(y_pred.shape)
    selection_c = copy(selection)
    selection_c[quantile_axis] = 0
    selection_c = tuple(selection_c)
    selection_r = copy(selection)
    selection_r[quantile_axis] = 1
    selection_r = tuple(selection_r)
    dy = y_pred[selection_r] - y_pred[selection_c]
    dy /= quantiles[1] - quantiles[0]
    x_cdf_l = y_pred[selection_c] - 2.0 * quantiles[0] * dy
    x_cdf_l = expand_dims(xp, x_cdf_l, quantile_axis)

    selection_l = copy(selection)
    selection_l[quantile_axis] = -2
    selection_l = tuple(selection_l)
    selection_c = copy(selection)
    selection_c[quantile_axis] = -1
    selection_c = tuple(selection_c)
    dy = y_pred[selection_c] - y_pred[selection_l]
    dy /= quantiles[-1] - quantiles[-2]
    x_cdf_r = y_pred[selection_c] + 2.0 * (1.0 - quantiles[-1]) * dy
    x_cdf_r = expand_dims(xp, x_cdf_r, quantile_axis)

    x_cdf = concatenate(xp, [x_cdf_l, y_pred, x_cdf_r], quantile_axis)

    selection_l = [slice(0, None)] * n_dims
    selection_l[quantile_axis] = slice(0, -1)
    selection_l = tuple(selection_l)
    selection_r = [slice(0, None)] * n_dims
    selection_r[quantile_axis] = slice(1, None)
    selection_r = tuple(selection_r)

    x_index = [slice(0, None)] * n_dims
    x_index[quantile_axis] = 0

    y_pdf_new = r(x_pdf, dist_axis=quantile_axis) * y_pdf

    selection = [slice(0, None)] * n_dims
    selection[quantile_axis] = slice(1, -1)
    selection = tuple(selection)
    y_cdf_new = cumtrapz(xp, y_pdf_new[selection], x_cdf, quantile_axis)

    selection = [slice(0, None)] * n_dims
    selection[quantile_axis] = slice(-1, None)
    selection = tuple(selection)
    y_cdf_new = y_cdf_new / y_cdf_new[selection]

    x_cdf_l = x_cdf[selection_l]
    x_cdf_r = x_cdf[selection_r]
    y_cdf_new_l = y_cdf_new[selection_l]
    y_cdf_new_r = y_cdf_new[selection_r]

    y_pred_new = []

    for i in range(0, len(quantiles)):
        q = quantiles[i]

        mask = as_type(xp, (y_cdf_new_l < q) * (y_cdf_new_r >= q), x_cdf_l)
        y_new = x_cdf_l * (y_cdf_new_r - q) * mask
        y_new += x_cdf_r * (q - y_cdf_new_l) * mask
        y_new /= mask * (y_cdf_new_r - y_cdf_new_l) + (1.0 - mask)
        y_new = expand_dims(xp, y_new.sum(quantile_axis), quantile_axis)

        y_pred_new.append(y_new)

    y_pred_new = concatenate(xp, y_pred_new, quantile_axis)
    return y_pred_new


def posterior_maximum(y_pred, quantiles, quantile_axis=1):
    if len(y_pred.shape) == 1:
        quantile_axis = 0
    xp = get_array_module(y_pred)

    x, y = pdf(y_pred, quantiles, quantile_axis=quantile_axis)
    indices = argmax(xp, y, axes=quantile_axis)
    shape = indices.shape
    indices = expand_dims(xp, indices, quantile_axis)
    return take_along_axis(xp, x, indices, axis=quantile_axis).reshape(shape)


