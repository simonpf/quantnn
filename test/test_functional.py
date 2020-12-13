import einops as eo
import numpy as np
import pytest
from quantnn.generic import sample_uniform, to_array
from quantnn.functional import (cdf, pdf, posterior_mean, crps,
                                probability_less_than,
                                probability_larger_than,
                                sample_posterior,
                                sample_posterior_gaussian,
                                quantile_loss)

backends = [np]
try:
    import torch
    backends.append(torch)
except Exception:
    pass

@pytest.mark.parametrize("xp", backends)
def test_cdf(xp):
    """
    Tests the calculation of the pdf for different shapes of input
    arrays.
    """

    #
    # 1D predictions
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = xp.arange(1.0, 9.1, 1.0)
    x_cdf, y_cdf = cdf(y_pred, quantiles)
    assert xp.all(xp.isclose(x_cdf[0], xp.zeros_like(x_cdf[0])))
    assert xp.all(xp.isclose(x_cdf[-1], 10.0 * xp.ones_like(x_cdf[-1])))

    #
    # 2D predictions
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> w q', w=10)
    x_cdf, y_cdf = cdf(y_pred, quantiles)
    assert xp.all(xp.isclose(x_cdf[:, 0], xp.zeros_like(x_cdf[:, 0])))
    assert xp.all(xp.isclose(x_cdf[:, -1], 10.0 * xp.ones_like(x_cdf[:, -1])))

    #
    # 3D predictions, quantiles along last axis
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> h w q', h=10, w=10)
    x_cdf, y_cdf = cdf(y_pred, quantiles, quantile_axis=-1)
    assert xp.all(xp.isclose(x_cdf[:, :, 0], xp.zeros_like(x_cdf[:, :, 0])))
    assert xp.all(xp.isclose(x_cdf[:, :, -1], 10.0 * xp.ones_like(x_cdf[:, :, -1])))

    #
    # 3D predictions, quantiles along first axis
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> h q w', h=10, w=10)
    x_cdf, y_cdf = cdf(y_pred, quantiles, quantile_axis=1)
    assert xp.all(xp.isclose(x_cdf[:, 0, :], xp.zeros_like(x_cdf[:, 0, :])))
    assert xp.all(xp.isclose(x_cdf[:, -1, :], 10.0 * xp.ones_like(x_cdf[:, -1, :])))

@pytest.mark.parametrize("xp", backends)
def test_pdf(xp):
    """
    Tests the calculation of the pdf for different shapes of input
    arrays.
    """

    #
    # 1D predictions
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = xp.arange(1.0, 9.1, 1.0)
    x_pdf, y_pdf = pdf(y_pred, quantiles)
    assert xp.all(xp.isclose(x_pdf[1:-1], xp.arange(0.5, 9.6, 1.0)))
    assert xp.all(xp.isclose(y_pdf[[0, -1]], xp.zeros_like(y_pdf[[0, -1]])))
    assert xp.all(xp.isclose(y_pdf[1:-1], 0.1 * xp.ones_like(y_pdf[1:-1])))

    #
    # 2D predictions
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> w q', w=10)
    x_pdf, y_pdf = pdf(y_pred, quantiles)
    assert xp.all(xp.isclose(x_pdf[:, 1:-1],
                             xp.arange(0.5, 9.6, 1.0).reshape(1, -1)))
    assert xp.all(xp.isclose(y_pdf[:, [0, -1]], xp.zeros_like(y_pdf[:, [0, -1]])))
    assert xp.all(xp.isclose(y_pdf[:, 1:-1], 0.1 * xp.ones_like(y_pdf[:, 1:-1])))

    #
    # 3D predictions, quantiles along last axis
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> h w q', h=10, w=10)
    x_pdf, y_pdf = pdf(y_pred, quantiles, quantile_axis=-1)
    assert xp.all(xp.isclose(x_pdf[:, :, 1:-1],
                             xp.arange(0.5, 9.6, 1.0).reshape(1, 1, -1)))
    assert xp.all(xp.isclose(y_pdf[:, :, [0, -1]], xp.zeros_like(y_pdf[:, :, [0, -1]])))
    assert xp.all(xp.isclose(y_pdf[:, :, 1:-1], 0.1 * xp.ones_like(y_pdf[:, :, 1:-1])))

    #
    # 3D predictions, quantiles along first axis
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> h q w', h=10, w=10)
    x_pdf, y_pdf = pdf(y_pred, quantiles, quantile_axis=1)
    assert xp.all(xp.isclose(x_pdf[:, 1:-1, :],
                             xp.arange(0.5, 9.6, 1.0).reshape(1, -1, 1)))
    assert xp.all(xp.isclose(y_pdf[:, [0, -1], :], xp.zeros_like(y_pdf[:, [0, -1], :])))
    assert xp.all(xp.isclose(y_pdf[:, 1:-1, :], 0.1 * xp.ones_like(y_pdf[:, 1:-1, :])))

@pytest.mark.parametrize("xp", backends)
def test_posterior_mean(xp):
    """
    Tests the calculation of the posterior mean for different shapes of
    input arrays.
    """

    #
    # 1D predictions
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = xp.arange(1.0, 9.1, 1.0)
    means = posterior_mean(y_pred, quantiles)
    assert xp.all(xp.isclose(means, 5.0 * xp.ones_like(means)))

    #
    # 2D predictions
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> w q', w=10)
    means = posterior_mean(y_pred, quantiles)
    assert xp.all(xp.isclose(means, 5.0 * xp.ones_like(means)))

    #
    # 3D predictions, quantiles along last axis
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> h w q', h=10, w=10)
    means = posterior_mean(y_pred, quantiles, quantile_axis=-1)
    assert xp.all(xp.isclose(means, 5.0 * xp.ones_like(means)))

    #
    # 3D predictions, quantiles along first axis
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> h q w', h=10, w=10)
    means = posterior_mean(y_pred, quantiles, quantile_axis=1)
    assert xp.all(xp.isclose(means, 5.0 * xp.ones_like(means)))

@pytest.mark.parametrize("xp", backends)
def test_crps(xp):
    """
    Tests the calculation of the CRPS for different shapes of input
    arrays.
    """

    #
    # 1D predictions
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = xp.arange(1.0, 9.1, 1.0)
    scores = crps(y_pred, quantiles, 4.9)
    assert xp.all(xp.isclose(scores, 0.85 * xp.ones_like(scores)))

    #
    # 2D predictions
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> w q', w=10)
    y_true = 4.9 * xp.ones(10)
    scores = crps(y_pred, quantiles, y_true)
    assert xp.all(xp.isclose(scores, 0.85 * xp.ones_like(scores)))

    ##
    ## 3D predictions, quantiles along last axis
    ##

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> h w q', w=10, h=10)
    y_true = 4.9 * xp.ones((10, 10))
    scores = crps(y_pred, quantiles, y_true, quantile_axis=2)
    assert xp.all(xp.isclose(scores, 0.85 * xp.ones_like(scores)))

    ##
    ## 3D predictions, quantiles along first axis
    ##

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> h q w', w=10, h=10)
    y_true = 4.9 * xp.ones((10, 10))
    scores = crps(y_pred, quantiles, y_true, quantile_axis=1)
    assert xp.all(xp.isclose(scores, 0.85 * xp.ones_like(scores)))

@pytest.mark.parametrize("xp", backends)
def test_probability_less_than(xp):
    """
    Tests predicting the probability that the true value is lower
    than a given threshold.
    """

    #
    # 1D predictions
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = xp.arange(1.0, 9.1, 1.0)
    t = 10.0 * np.random.rand()
    probability = probability_less_than(y_pred, quantiles, t)
    assert xp.all(xp.isclose(probability, 0.1 * t * xp.ones_like(probability)))

    #
    # 2D predictions
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> w q', w=10)
    t = 10.0 * np.random.rand()
    probability = probability_less_than(y_pred, quantiles, t)
    assert xp.all(xp.isclose(probability, 0.1 * t * xp.ones_like(probability)))

    #
    # 3D predictions, quantiles along last axis
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> h w q', h=10, w=10)
    t = 10.0 * np.random.rand()
    probability = probability_less_than(y_pred, quantiles, t, quantile_axis=2)
    assert xp.all(xp.isclose(probability, 0.1 * t * xp.ones_like(probability)))

    #
    # 3D predictions, quantiles along first axis
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> h q w', h=10, w=10)
    t = 10.0 * np.random.rand()
    probability = probability_less_than(y_pred, quantiles, t, quantile_axis=1)
    assert xp.all(xp.isclose(probability, 0.1 * t * xp.ones_like(probability)))

@pytest.mark.parametrize("xp", backends)
def test_probability_larger_than(xp):
    """
    Tests predicting the probability that the true value is larger
    than a given threshold.
    """

    #
    # 1D predictions
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = xp.arange(1.0, 9.1, 1.0)
    t = 10.0 * np.random.rand()
    probability = probability_larger_than(y_pred, quantiles, t)
    assert xp.all(xp.isclose(probability, 1.0 - 0.1 * t * xp.ones_like(probability)))

    #
    # 2D predictions
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> w q', w=10)
    t = 10.0 * np.random.rand()
    probability = probability_larger_than(y_pred, quantiles, t)
    assert xp.all(xp.isclose(probability, 1.0 - 0.1 * t * xp.ones_like(probability)))

    #
    # 3D predictions, quantiles along last axis
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> h w q', h=10, w=10)
    t = 10.0 * np.random.rand()
    probability = probability_larger_than(y_pred, quantiles, t, quantile_axis=2)
    assert xp.all(xp.isclose(probability, 1.0 - 0.1 * t * xp.ones_like(probability)))

    #
    # 3D predictions, quantiles along first axis
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> h q w', h=10, w=10)
    t = 10.0 * np.random.rand()
    probability = probability_larger_than(y_pred, quantiles, t, quantile_axis=1)
    assert xp.all(xp.isclose(probability, 1.0 - 0.1 * t * xp.ones_like(probability)))

@pytest.mark.parametrize("xp", backends)
def test_sample_posterior(xp):
    """
    Tests sampling from the posterior by interpolation of inverse CDF.
    """

    #
    # 1D predictions
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = xp.arange(1.0, 9.1, 1.0)
    samples = sample_posterior(y_pred, quantiles, n_samples=10000)
    assert xp.all(xp.isclose(samples.mean(),
                             5.0 * xp.ones_like(samples.mean()),
                             1e-1))

    #
    # 2D predictions
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> w q', w=10)
    samples = sample_posterior(y_pred, quantiles, n_samples=10000)
    assert xp.all(xp.isclose(samples.mean(),
                             5.0 * xp.ones_like(samples.mean()),
                             1e-1))

    #
    # 3D predictions, quantiles along last axis
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> h w q', h=10, w=10)
    samples = sample_posterior(y_pred, quantiles, n_samples=1000,
                               quantile_axis=2)
    assert xp.all(xp.isclose(samples.mean(),
                             5.0 * xp.ones_like(samples.mean()),
                             1e-1))

    #
    # 3D predictions, quantiles along first axis
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> h q w', h=10, w=10)
    samples = sample_posterior(y_pred, quantiles, n_samples=1000,
                               quantile_axis=1)
    assert xp.all(xp.isclose(samples.mean(),
                             5.0 * xp.ones_like(samples.mean()),
                             1e-1))

@pytest.mark.parametrize("xp", backends)
def test_sample_posterior_gaussian(xp):
    """
    Tests sampling from the posterior by fitting a Gaussian.
    """

    #
    # 1D predictions
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = xp.arange(1.0, 9.1, 1.0)
    samples = sample_posterior_gaussian(y_pred, quantiles, n_samples=10000)
    assert xp.all(xp.isclose(samples.mean(),
                             5.0 * xp.ones_like(samples.mean()),
                             1e-1))

    #
    # 2D predictions
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> w q', w=10)
    samples = sample_posterior_gaussian(y_pred, quantiles, n_samples=10000)
    assert xp.all(xp.isclose(samples.mean(),
                             5.0 * xp.ones_like(samples.mean()),
                             1e-1))

    #
    # 3D predictions, quantiles along last axis
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> h w q', h=10, w=10)
    samples = sample_posterior_gaussian(y_pred, quantiles, n_samples=1000,
                               quantile_axis=2)
    assert xp.all(xp.isclose(samples.mean(),
                             5.0 * xp.ones_like(samples.mean()),
                             1e-1))

    #
    # 3D predictions, quantiles along first axis
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> h q w', h=10, w=10)
    samples = sample_posterior_gaussian(y_pred, quantiles, n_samples=1000,
                               quantile_axis=1)
    assert xp.all(xp.isclose(samples.mean(),
                             5.0 * xp.ones_like(samples.mean()),
                             1e-1))

@pytest.mark.parametrize("xp", backends)
def test_quantile_loss(xp):
    """
    Tests calculation of the quantile loss function.
    """

    #
    # 1D predictions
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = xp.arange(1.0, 9.1, 1.0)
    y_true = to_array(xp, [5.0])
    loss = quantile_loss(y_pred, quantiles, y_true)
    assert xp.isclose(loss.mean(), to_array(xp, [0.444444]))

    #
    # 2D predictions
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> w q', w=10)
    y_true = eo.repeat(to_array(xp, [5.0]), 'q -> w q', w=10)
    loss = quantile_loss(y_pred, quantiles, y_true)
    assert xp.isclose(loss.mean(), to_array(xp, [0.444444]))

    #
    # 3D predictions, quantiles along last axis
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> h w q', h=10, w=10)
    y_true = eo.repeat(to_array(xp, [5.0]), 'q -> h w q', h=10, w=10)
    loss = quantile_loss(y_pred, quantiles,  y_true, quantile_axis=-1)
    assert xp.isclose(loss.mean(), to_array(xp, [0.444444]))

    #
    # 3D predictions, quantiles along first axis
    #

    quantiles = xp.arange(0.1, 0.91, 0.1)
    y_pred = eo.repeat(xp.arange(1.0, 9.1, 1.0), 'q -> h q w', h=10, w=10)
    y_true = eo.repeat(to_array(xp, [5.0]), 'q -> h q w', h=10, w=10)
    loss = quantile_loss(y_pred, quantiles, y_true, quantile_axis=1)
    assert xp.isclose(loss.mean(), to_array(xp, [0.444444]))
