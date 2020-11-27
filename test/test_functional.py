import numpy as np
from qrnn.functional import cdf, pdf, posterior_mean

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
    Tests the calculation of the pdf for different shapes of input
    arrays.
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
