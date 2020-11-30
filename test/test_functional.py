import numpy as np
from quantnn.functional import (cdf, pdf, posterior_mean, crps,
                                probability_less_than,
                                probability_larger_than,
                                sample_posterior,
                                sample_posterior_gaussian,
                                quantile_loss)

def test_cdf():
    """
    Tests the calculation of the pdf for different shapes of input
    arrays.
    """

    #
    # 1D predictions
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.arange(1.0, 9.1, 1.0)
    x_cdf, y_cdf = cdf(y_pred, quantiles)
    assert np.all(np.isclose(x_cdf[0], 0.0))
    assert np.all(np.isclose(x_cdf[-1], 10.0))

    #
    # 2D predictions
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0), (10, 1))
    x_cdf, y_cdf = cdf(y_pred, quantiles)
    assert np.all(np.isclose(x_cdf[:, 0], 0.0))
    assert np.all(np.isclose(x_cdf[:, -1], 10.0))

    #
    # 3D predictions, quantiles along last axis
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0), (10, 10, 1))
    x_cdf, y_cdf = cdf(y_pred, quantiles, quantile_axis=-1)
    assert np.all(np.isclose(x_cdf[:, :, 0], 0.0))
    assert np.all(np.isclose(x_cdf[:, :, -1], 10.0))

    #
    # 3D predictions, quantiles along first axis
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0).reshape(-1, 1), (10, 1, 10))
    x_cdf, y_cdf = cdf(y_pred, quantiles, quantile_axis=1)
    assert np.all(np.isclose(x_cdf[:, 0, :], 0.0))
    assert np.all(np.isclose(x_cdf[:, -1, :], 10.0))

def test_pdf():
    """
    Tests the calculation of the pdf for different shapes of input
    arrays.
    """

    #
    # 1D predictions
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.arange(1.0, 9.1, 1.0)
    x_pdf, y_pdf = pdf(y_pred, quantiles)
    assert np.all(np.isclose(x_pdf[1:-1], np.arange(0.5, 9.6, 1.0)))
    assert np.all(np.isclose(y_pdf[[0, -1]], 0.0))
    assert np.all(np.isclose(y_pdf[1:-1], 0.1))

    #
    # 2D predictions
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0), (10, 1))
    x_pdf, y_pdf = pdf(y_pred, quantiles)
    assert np.all(np.isclose(x_pdf[:, 1:-1],
                             np.arange(0.5, 9.6, 1.0).reshape(1, -1)))
    assert np.all(np.isclose(y_pdf[:, 0], 0.0))
    assert np.all(np.isclose(y_pdf[:, -1], 0.0))
    assert np.all(np.isclose(y_pdf[:, 1:-1], 0.1))

    #
    # 3D predictions, quantiles along last axis
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0), (10, 10, 1))
    x_pdf, y_pdf = pdf(y_pred, quantiles, quantile_axis=-1)
    assert np.all(np.isclose(x_pdf[:, :, 1:-1],
                             np.arange(0.5, 9.6, 1.0).reshape(1, 1, -1)))
    assert np.all(np.isclose(y_pdf[:, :, 1:-1], 0.1))
    assert np.all(np.isclose(y_pdf[:, :, 0], 0.0))
    assert np.all(np.isclose(y_pdf[:, :, -1], 0.0))

    #
    # 3D predictions, quantiles along first axis
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0).reshape(-1, 1), (10, 1, 10))
    x_pdf, y_pdf = pdf(y_pred, quantiles, quantile_axis=1)
    assert np.all(np.isclose(x_pdf[:, 1:-1, :],
                             np.arange(0.5, 9.6, 1.0).reshape(1, -1, 1)))
    assert np.all(np.isclose(y_pdf[:, 1:-1, :], 0.1))
    assert np.all(np.isclose(y_pdf[:, 0, :], 0.0))
    assert np.all(np.isclose(y_pdf[:, -1, :], 0.0))

def test_posterior_mean():
    """
    Tests the calculation of the posterior mean for different shapes of
    input arrays.
    """

    #
    # 1D predictions
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.arange(1.0, 9.1, 1.0)
    means = posterior_mean(y_pred, quantiles)
    assert np.all(np.isclose(means, 5.0))

    #
    # 2D predictions
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0), (10, 1))
    means = posterior_mean(y_pred, quantiles)
    assert np.all(np.isclose(means, 5.0))

    #
    # 3D predictions, quantiles along last axis
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0), (10, 10, 1))
    means = posterior_mean(y_pred, quantiles, quantile_axis=-1)
    assert np.all(np.isclose(means, 5.0))

    #
    # 3D predictions, quantiles along first axis
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0).reshape(-1, 1), (10, 1, 10))
    means = posterior_mean(y_pred, quantiles, quantile_axis=1)
    assert np.all(np.isclose(means, 5.0))

def test_crps():
    """
    Tests the calculation of the CRPS for different shapes of input
    arrays.
    """

    #
    # 1D predictions
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.arange(1.0, 9.1, 1.0)
    scores = crps(y_pred, quantiles, 4.9)
    assert np.all(np.isclose(scores, 0.85))

    ##
    ## 2D predictions
    ##

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0), (10, 1))
    y_true = 4.9 * np.ones(10)
    scores = crps(y_pred, quantiles, y_true)
    assert np.all(np.isclose(scores, 0.85))

    ##
    ## 3D predictions, quantiles along last axis
    ##

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0), (10, 10, 1))
    y_true = 4.9 * np.ones((10, 10))
    scores = crps(y_pred, quantiles, y_true, quantile_axis=2)
    assert np.all(np.isclose(scores, 0.85))

    ##
    ## 3D predictions, quantiles along first axis
    ##

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0).reshape(-1, 1), (10, 1, 10))
    y_true = 4.9 * np.ones((10, 10))
    scores = crps(y_pred, quantiles, y_true, quantile_axis=1)
    assert np.all(np.isclose(scores, 0.85))

def test_probability_less_than():
    """
    Tests predicting the probability that the true value is lower
    than a given threshold.
    """

    #
    # 1D predictions
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.arange(1.0, 9.1, 1.0)
    t = np.random.rand() * 10.0
    probability = probability_less_than(y_pred, quantiles, t)
    assert np.all(np.isclose(probability, 0.1 * t))

    #
    # 2D predictions
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0), (10, 1))
    t = np.random.rand() * 10.0
    probability = probability_less_than(y_pred, quantiles, t)
    assert np.all(np.isclose(probability, 0.1 * t))

    #
    # 3D predictions, quantiles along last axis
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0), (10, 10, 1))
    t = np.random.rand() * 10.0
    probability = probability_less_than(y_pred, quantiles, t, quantile_axis=2)
    assert np.all(np.isclose(probability, 0.1 * t))

    #
    # 3D predictions, quantiles along first axis
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0).reshape(-1, 1), (10, 1, 10))
    t = np.random.rand() * 10.0
    probability = probability_less_than(y_pred, quantiles, t, quantile_axis=1)
    assert np.all(np.isclose(probability, 0.1 * t))

def test_probability_larger_than():
    """
    Tests predicting the probability that the true value is larger
    than a given threshold.
    """

    #
    # 1D predictions
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.arange(1.0, 9.1, 1.0)
    t = np.random.rand() * 10.0
    probability = probability_larger_than(y_pred, quantiles, t)
    assert np.all(np.isclose(probability, 1.0 - 0.1 * t))

    #
    # 2D predictions
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0), (10, 1))
    t = np.random.rand() * 10.0
    probability = probability_larger_than(y_pred, quantiles, t)
    assert np.all(np.isclose(probability, 1.0 - 0.1 * t))

    #
    # 3D predictions, quantiles along last axis
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0), (10, 10, 1))
    t = np.random.rand() * 10.0
    probability = probability_larger_than(y_pred, quantiles, t, quantile_axis=2)
    assert np.all(np.isclose(probability, 1.0 - 0.1 * t))

    #
    # 3D predictions, quantiles along first axis
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0).reshape(-1, 1), (10, 1, 10))
    t = np.random.rand() * 10.0
    probability = probability_larger_than(y_pred, quantiles, t, quantile_axis=1)
    assert np.all(np.isclose(probability, 1.0 - 0.1 * t))

def test_sample_posterior():
    """
    Tests sampling from the posterior by interpolation of inverse CDF.
    """

    #
    # 1D predictions
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.arange(1.0, 9.1, 1.0)
    samples = sample_posterior(y_pred, quantiles, n_samples=10000)
    assert np.all(np.isclose(samples.mean(), 5.0, 1e-1))

    #
    # 2D predictions
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0), (10, 1))
    samples = sample_posterior(y_pred, quantiles, n_samples=10000)
    assert np.all(np.isclose(samples.mean(), 5.0, 1e-1))

    #
    # 3D predictions, quantiles along last axis
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0), (10, 10, 1))
    samples = sample_posterior(y_pred, quantiles, n_samples=1000,
                               quantile_axis=2)
    assert np.all(np.isclose(samples.mean(), 5.0, 1e-1))

    #
    # 3D predictions, quantiles along first axis
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0).reshape(-1, 1), (10, 1, 10))
    samples = sample_posterior(y_pred, quantiles, n_samples=1000,
                               quantile_axis=1)
    assert np.all(np.isclose(samples.mean(), 5.0, 1e-1))

def test_sample_posterior_gaussian_fit():
    """
    Tests sampling from the posterior by fitting a Gaussian.
    """

    #
    # 1D predictions
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.arange(1.0, 9.1, 1.0)
    samples = sample_posterior_gaussian(y_pred, quantiles, n_samples=10000)
    assert np.all(np.isclose(samples.mean(), 5.0, 1e-1))

    #
    # 2D predictions
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0), (10, 1))
    samples = sample_posterior_gaussian(y_pred, quantiles, n_samples=10000)
    assert np.all(np.isclose(samples.mean(), 5.0, 1e-1))

    #
    # 3D predictions, quantiles along last axis
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0), (10, 10, 1))
    samples = sample_posterior_gaussian(y_pred, quantiles, n_samples=1000,
                               quantile_axis=2)
    assert np.all(np.isclose(samples.mean(), 5.0, 1e-1))

    #
    # 3D predictions, quantiles along first axis
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0).reshape(-1, 1), (10, 1, 10))
    samples = sample_posterior_gaussian(y_pred, quantiles, n_samples=1000,
                               quantile_axis=1)
    assert np.all(np.isclose(samples.mean(), 5.0, 1e-1))

def test_quantile_loss():
    """
    Tests calculation of the quantile loss function.
    """

    #
    # 1D predictions
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.arange(1.0, 9.1, 1.0)
    y_true = np.array([5.0])
    loss = quantile_loss(y_pred, quantiles, y_true)
    assert np.isclose(loss.mean(), 0.444444)

    #
    # 2D predictions
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0), (10, 1))
    y_true = np.tile(np.array([5.0]), (10, 1))
    loss = quantile_loss(y_pred, quantiles, y_true)
    assert np.isclose(loss.mean(), 0.444444)

    #
    # 3D predictions, quantiles along last axis
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0), (10, 10, 1))
    y_true = np.tile(np.array([5.0]), (10, 10, 1))
    loss = quantile_loss(y_pred, quantiles,  y_true, quantile_axis=-1)
    assert np.isclose(loss.mean(), 0.444444)

    #
    # 3D predictions, quantiles along first axis
    #

    quantiles = np.arange(0.1, 0.91, 0.1)
    y_pred = np.tile(np.arange(1.0, 9.1, 1.0).reshape(-1, 1), (10, 1, 10))
    y_true = np.tile(np.array([5.0]), (10, 1, 10))
    loss = quantile_loss(y_pred, quantiles, y_true, quantile_axis=1)
    assert np.isclose(loss.mean(), 0.444444)
