"""
qrnn.functional
===============

The ``qrnn.functional`` module provides functions to calculate relevant
 statistics from QRNN output.
"""
from copy import copy

import numpy as np
from scipy.stats import norm

from qrnn.common import (InvalidDimensionException,
                         get_array_module,
                         to_array,
                         sample_uniform,
                         sample_gaussian)


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
    xp = get_array_module(y_pred)

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
    dy = (x_cdf[selection_r] - x_cdf[selection_c])
    dy /= (quantiles[1] - quantiles[0])
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
    dy = (x_cdf[selection_c] - x_cdf[selection_l])
    dy /= (quantiles[-1] - quantiles[-2])
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

    xp = get_array_module(y_pred)

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
    xp = get_array_module(y_pred)

    x_cdf, y_cdf = cdf(y_pred, quantiles, quantile_axis=quantile_axis)
    return xp.trapz(x_cdf, x=y_cdf, axis=quantile_axis)

def crps(y_pred, quantiles, y_true, quantile_axis=1):
    r"""
    Compute the Continuous Ranked Probability Score (CRPS) for given
    predicted quantiles.

    This function uses a piece-wise linear fit to the approximate posterior
    CDF obtained from the predicted quantiles in :code:`y_pred` to
    approximate the continuous ranked probability score (CRPS):

    .. math::
        CRPS(\mathbf{y}, x) = \int_{-\infty}^\infty (F_{x | \mathbf{y}}(x')
        - \mathrm{1}_{x < x'})^2 \: dx'

    Args::

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
    y_true = np.asarray(y_true)
    y_true = y_true.reshape(y_true_shape)

    ind = xp.zeros(x_cdf.shape)
    ind[x_cdf > y_true] = 1.0

    output_shape = list(x_cdf.shape)
    del output_shape[quantile_axis]
    integral = xp.zeros(output_shape)
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
    probabilities = xp.zeros(output_shape)

    y_l = y_cdf[0]
    x_index = [slice(0, None)] * n_dims
    x_index[quantile_axis] = 0
    x_l = x_cdf[tuple(x_index)]

    for i in range(1, len(y_cdf)):
        y_r = y_cdf[i]
        x_index[quantile_axis] = i
        x_r = x_cdf[tuple(x_index)]

        inds = np.logical_and(x_l < y, x_r >= y)
        probabilities[inds] = y_l * (x_r[inds] - y)
        probabilities[inds] += y_r * (y - x_l[inds])
        probabilities[inds] /= (x_r[inds] - x_l[inds])

        y_l = y_r
        x_l = x_r

    return probabilities

def probability_larger_than(y_pred,
                            quantiles,
                            y,
                            quantile_axis=1):
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
    return 1.0 - probability_less_than(y_pred,
                                       quantiles,
                                       y,
                                       quantile_axis=quantile_axis)


def sample_posterior(y_pred,
                     quantiles,
                     n_samples=1,
                     quantile_axis=1):
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

    samples = sample_uniform(xp, tuple(output_shape))
    results = xp.zeros(samples.shape)

    y_l = y_cdf[0]
    x_index = [slice(0, None)] * n_dims
    x_index[quantile_axis] = slice(0, 1)
    x_l = x_cdf[tuple(x_index)]

    for i in range(1, len(y_cdf)):
        y_r = y_cdf[i]
        x_index[quantile_axis] = slice(i, i + 1)
        x_r = x_cdf[tuple(x_index)]

        inds = np.logical_and(samples > y_l, samples <= y_r)
        results[inds] = (x_l * (y_r - samples))[inds]
        results[inds] += (x_r * (samples - y_l))[inds]
        results[inds] /= (y_r - y_l)

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
    x_shape = [1,] * n_dims
    x_shape[quantile_axis] = -1
    x_shape = tuple(x_shape)
    x = x.reshape(x_shape)

    output_shape = list(y_pred.shape)
    output_shape[quantile_axis] = 1
    output_shape = tuple(output_shape)

    d2e_00 = x.size
    d2e_01 = x.sum()
    d2e_10 = x.sum()
    d2e_11 = np.sum(x ** 2)

    d2e_det_inv = 1.0 / (d2e_00 * d2e_11 - d2e_01 * d2e_11)
    d2e_inv_00 = d2e_det_inv * d2e_11
    d2e_inv_01 = -d2e_det_inv * d2e_01
    d2e_inv_10 = -d2e_det_inv * d2e_10
    d2e_inv_11 = d2e_det_inv * d2e_00

    x = x.reshape(x_shape)
    de_0 = -np.sum(y_pred - x, axis=quantile_axis).reshape(output_shape)
    de_1 = -np.sum(x * (y_pred - x), axis=quantile_axis).reshape(output_shape)

    mu = -(d2e_inv_00 * de_0 + d2e_inv_01 * de_1)
    sigma = 1.0 - (d2e_inv_10 * de_0 + d2e_inv_11 * de_1)

    return mu, sigma

def sample_posterior_gaussian(y_pred,
                              quantiles,
                              n_samples=1,
                              quantile_axis=1):
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
    mu, sigma = fit_gaussian_to_quantiles(y_pred,
                                          quantiles,
                                          quantile_axis=quantile_axis)
    print(mu.shape)

    output_shape = list(y_pred.shape)
    output_shape[quantile_axis] = n_samples
    samples = sample_gaussian(xp, tuple(output_shape))
    return mu + sigma * samples
