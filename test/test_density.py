"""
Tests for the quantnn.density module.
"""
import einops as eo
import numpy as np
import pytest

from quantnn.generic import sample_uniform, to_array, arange, reshape
from quantnn.density import (posterior_quantiles,
                             posterior_cdf,
                             posterior_mean,
                             probability_larger_than,
                             probability_less_than,
                             sample_posterior)

@pytest.mark.parametrize("xp", pytest.backends)
def test_posterior_cdf(xp):
    """
    Test calculation and normalization of posterior cdf.
    """

    #
    # 1D predictions
    #

    bins = arange(xp, 0.0, 10.1, 0.1)
    y_pred = 0.1 * xp.ones(100)

    y_cdf = posterior_cdf(y_pred, bins)

    assert y_cdf[-1] == 1.0
    assert y_cdf[0] == 0.0

    #
    # 2D predictions
    #

    bins = arange(xp, 0.0, 10.1, 0.1)
    y_pred = 0.1 * xp.ones(100)
    y_pred = eo.repeat(y_pred, 'q -> w q', w=10)

    y_cdf = posterior_cdf(y_pred, bins)

    assert np.all(np.isclose(y_cdf[:, 0], 0.0))
    assert np.all(np.isclose(y_cdf[:, -1], 1.0))
    assert y_cdf.shape[0] == 10
    assert y_cdf.shape[1] == 101

    y_pred = 0.1 * xp.ones(100)
    y_pred = eo.repeat(y_pred, 'q -> q w', w=10)

    y_cdf = posterior_cdf(y_pred, bins, bin_axis=0)

    assert np.all(np.isclose(y_cdf[0, :], 0.0))
    assert np.all(np.isclose(y_cdf[-1, :], 1.0))
    assert y_cdf.shape[0] == 101
    assert y_cdf.shape[1] == 10

    #
    # 3D predictions
    #

    bins = arange(xp, 0.0, 10.1, 0.1)
    y_pred = 0.1 * xp.ones(100)
    y_pred = eo.repeat(y_pred, 'q -> v w q', v=10, w=10)

    y_cdf = posterior_cdf(y_pred, bins, bin_axis=-1)

    assert np.all(np.isclose(y_cdf[:, :, 0], 0.0))
    assert np.all(np.isclose(y_cdf[:, :, -1], 1.0))
    assert y_cdf.shape[0] == 10
    assert y_cdf.shape[-1] == 101

    y_pred = 0.1 * xp.ones(100)
    y_pred = eo.repeat(y_pred, 'q -> q v w', v=10, w=10)

    y_cdf = posterior_cdf(y_pred, bins, bin_axis=0)

    assert np.all(np.isclose(y_cdf[0, :, :], 0.0))
    assert np.all(np.isclose(y_cdf[-1, :, :], 1.0))
    assert y_cdf.shape[0] == 101
    assert y_cdf.shape[-1] == 10

@pytest.mark.parametrize("xp", pytest.backends)
def test_posterior_quantiles(xp):
    """
    Tests the calculation of the pdf for different shapes of input
    arrays.
    """

    #
    # 1D predictions
    #

    bins = arange(xp, 1.0, 11.1, 0.1)
    y_pred = 0.1 * xp.ones(100)
    quantiles = arange(xp, 0.1, 0.91, 0.1)

    y_q = posterior_quantiles(y_pred, bins, quantiles)

    assert np.isclose(y_q[0], 2.0)
    assert np.isclose(y_q[-1], 10.0)

    #
    # 2D predictions
    #

    y_pred = 0.1 * xp.ones(100)
    y_pred = eo.repeat(y_pred, 'q -> w q', w=10)

    y_q = posterior_quantiles(y_pred, bins, quantiles)

    assert np.all(np.isclose(y_q[:, 0], 2.0))
    assert np.all(np.isclose(y_q[:, -1], 10.0))
    assert y_q.shape[0] == 10
    assert y_q.shape[1] == 9

    y_pred = 0.1 * xp.ones(100)
    y_pred = eo.repeat(y_pred, 'q -> q w', w=10)

    y_q = posterior_quantiles(y_pred, bins, quantiles, bin_axis=0)

    assert np.all(np.isclose(y_q[0, :], 2.0))
    assert np.all(np.isclose(y_q[-1, :], 10.0))
    assert y_q.shape[0] == 9
    assert y_q.shape[1] == 10

    #
    # 3D predictions
    #

    y_pred = 0.1 * xp.ones(100)
    y_pred = eo.repeat(y_pred, 'q -> v w q', v=10, w=10)

    y_q = posterior_quantiles(y_pred, bins, quantiles, bin_axis=-1)

    assert np.all(np.isclose(y_q[:, :, 0], 2.0))
    assert np.all(np.isclose(y_q[:, :, -1], 10.0))
    assert y_q.shape[0] == 10
    assert y_q.shape[-1] == 9

    y_pred = 0.1 * xp.ones(100)
    y_pred = eo.repeat(y_pred, 'q -> q v w', v=10, w=10)

    y_q = posterior_quantiles(y_pred, bins, quantiles, bin_axis=0)

    assert np.all(np.isclose(y_q[0, :, :], 2.0))
    assert np.all(np.isclose(y_q[-1, :, :], 10.0))
    assert y_q.shape[0] == 9
    assert y_q.shape[-1] == 10

@pytest.mark.parametrize("xp", pytest.backends)
def test_posterior_mean(xp):
    """
    Tests the calculation of the pdf for different shapes of input
    arrays.
    """

    #
    # 1D predictions
    #

    bins = arange(xp, 1.0, 11.1, 0.1)
    y_pred = 0.1 * xp.ones(100)

    mean = posterior_mean(y_pred, bins)

    assert np.isclose(mean, 6.0)

    #
    # 2D predictions
    #

    y_pred = 0.1 * xp.ones(100)
    y_pred = eo.repeat(y_pred, 'q -> w q', w=10)

    mean = posterior_mean(y_pred, bins)
    assert np.all(np.isclose(mean, 6.0))

    y_pred = 0.1 * xp.ones(100)
    y_pred = eo.repeat(y_pred, 'q -> q w', w=10)

    mean = posterior_mean(y_pred, bins, bin_axis=0)
    assert np.all(np.isclose(mean, 6.0))

    #
    # 3D predictions
    #

    y_pred = 0.1 * xp.ones(100)
    y_pred = eo.repeat(y_pred, 'q -> v w q', v=10, w=10)

    mean = posterior_mean(y_pred, bins, bin_axis=-1)
    assert np.all(np.isclose(mean, 6.0))

    y_pred = 0.1 * xp.ones(100)
    y_pred = eo.repeat(y_pred, 'q -> q v w', v=10, w=10)
    mean = posterior_mean(y_pred, bins, bin_axis=0)
    assert np.all(np.isclose(mean, 6.0))

@pytest.mark.parametrize("xp", pytest.backends)
def test_probability_less_and_larger_than(xp):
    """
    Tests the calculation of the pdf for different shapes of input
    arrays.
    """

    #
    # 1D predictions
    #

    bins = arange(xp, 1.0, 11.1, 0.1)
    y_pred = 0.1 * xp.ones(100)

    p = probability_less_than(y_pred, bins, 6.0)
    assert np.isclose(p, 0.5)
    p = probability_larger_than(y_pred, bins, 6.0)
    assert np.isclose(p, 0.5)

    #
    # 2D predictions
    #

    y_pred = 0.1 * xp.ones(100)
    y_pred = eo.repeat(y_pred, 'q -> w q', w=10)

    p = probability_less_than(y_pred, bins, 6.0)
    assert np.all(np.isclose(p, 0.5))

    y_pred = 0.1 * xp.ones(100)
    y_pred = eo.repeat(y_pred, 'q -> q w', w=10)

    p = probability_less_than(y_pred, bins, 6.0, bin_axis=0)
    assert np.all(np.isclose(p, 0.5))

    p = probability_larger_than(y_pred, bins, 6.0, bin_axis=0)
    assert np.all(np.isclose(p, 0.5))

    #
    # 3D predictions
    #

    y_pred = 0.1 * xp.ones(100)
    y_pred = eo.repeat(y_pred, 'q -> v w q', v=10, w=10)

    p = probability_less_than(y_pred, bins, 6.0, bin_axis=-1)
    assert np.all(np.isclose(p, 0.5))

    p = probability_larger_than(y_pred, bins, 6.0, bin_axis=-1)
    assert np.all(np.isclose(p, 0.5))

    y_pred = 0.1 * xp.ones(100)
    y_pred = eo.repeat(y_pred, 'q -> q v w', v=10, w=10)

    p = probability_less_than(y_pred, bins, 6.0, bin_axis=0)
    assert np.all(np.isclose(p, 0.5))

    p = probability_larger_than(y_pred, bins, 6.0, bin_axis=0)
    assert np.all(np.isclose(p, 0.5))

@pytest.mark.parametrize("xp", pytest.backends)
def test_sample_posterior(xp):
    """
    Tests the calculation of the pdf for different shapes of input
    arrays.
    """

    #
    # 1D predictions
    #

    bins = arange(xp, 1.0, 11.1, 0.1)
    y_pred = 0.1 * xp.ones(100)

    samples = sample_posterior(y_pred, bins, n_samples=100000)
    assert samples[0] >= 1.0
    assert samples[0] <= 11.0
    assert np.isclose(samples.mean(), 6.0, rtol=1e-1)

    #
    # 2D predictions
    #

    y_pred = 0.1 * xp.ones(100)
    y_pred = eo.repeat(y_pred, 'q -> w q', w=10)

    samples = sample_posterior(y_pred, bins, n_samples=10000)
    assert np.isclose(samples.mean(), 6.0, rtol=1e-1)

    #
    # 3D predictions, bins along second dimension
    #

    y_pred = 0.1 * xp.ones(100)
    y_pred = eo.repeat(y_pred, 'q -> w q h', w=10, h=10)

    samples = sample_posterior(y_pred, bins, n_samples=1000)
    assert np.isclose(samples.mean(), 6.0, rtol=1e-1)


    #
    # 3D predictions, bins along last dimension
    #

    y_pred = 0.1 * xp.ones(100)
    y_pred = eo.repeat(y_pred, 'q -> w h q', w=10, h=10)

    samples = sample_posterior(y_pred, bins, n_samples=1000, bin_axis=-1)
    assert np.isclose(samples.mean(), 6.0, rtol=1e-1)
