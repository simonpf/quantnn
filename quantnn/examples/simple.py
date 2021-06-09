"""
quantnn.examples.simple
=======================

This module provides a simple toy example to illustrate the basic
 functionality of quantile regression neural networks. The task is a simple
1-dimensional regression problem of a signal with heteroscedastic noise:

.. math::

  y = \sin(x) + \cdot \cos(x) \cdot \mathcal{N}(0, 1)

"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.cm import magma
from matplotlib.colors import Normalize


def create_training_data(n=1_000_000):
    """
    Create training data by randomly sampling the range :math:`[-\pi, \pi]`
    and computing y.

    Args:
        n(int): How many sample to compute.

    Return:
         Tuple ``(x, y)`` containing the input samples ``x`` given as 2D array
         with samples along first and input features along second dimension
         and the corresponding :math:`y` values in ``y``.
    """
    x = 2.0 * np.pi * np.random.random(size=n) - np.pi
    y = np.sin(x) + 1.0 * np.cos(x) * np.random.randn(n)
    return x, y


def create_validation_data(x):
    """
    Creates validation data for the toy example.

    In contrast to the generation of the training data this function allows
    specifying the x value of the data which allows plotting the predicted
    result over an arbitrary domain.

    Args:
        x: Arbitrary array containing the x values for which to compute
           corresponding y values.
    Return:
        Numpy array containing the y values corresponding to the given x
        values.
    """
    y = np.sin(x) + 1.0 * np.cos(x) * np.random.randn(*x.shape)
    return y


def plot_histogram(x, y):
    """
    Plot 2D histogram of data.
    """
    # Calculate histogram
    bins_x = np.linspace(-np.pi, np.pi, 201)
    bins_y = np.linspace(-4, 4, 201)
    x_img, y_img = np.meshgrid(bins_x, bins_y)
    img, _, _ = np.histogram2d(x, y, bins=(bins_x, bins_y), density=True)

    # Plot results
    f, ax = plt.subplots(1, 1, figsize=(10, 6))
    m = ax.pcolormesh(x_img, y_img, img.T, vmin=0, vmax=0.3, cmap="magma")
    x_sin = np.linspace(-np.pi, np.pi, 1001)
    y_sin = np.sin(x_sin)
    ax.plot(x_sin, y_sin, c="grey", label="$y=\sin(x)$", lw=3)
    ax.set_ylim([-2, 2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(m, label="Normalized frequency")
    plt.legend()


def plot_results(x_train, y_train, x_val, y_pred, y_mean, quantiles):
    """
    Plots the predicted quantiles against empirical quantiles.
    """
    # Calculate histogram and empirical quantiles.
    bins_x = np.linspace(-np.pi, np.pi, 201)
    bins_y = np.linspace(-4, 4, 201)
    x_img, y_img = np.meshgrid(bins_x, bins_y)
    img, _, _ = np.histogram2d(x_train, y_train, bins=(bins_x, bins_y), density=True)
    norm = np.trapz(img, x=0.5 * (bins_y[1:] + bins_y[:-1]), axis=1)
    img_normed = img / norm.reshape(-1, 1)
    img_cdf = sp.integrate.cumtrapz(
        img_normed, x=0.5 * (bins_y[1:] + bins_y[:-1]), axis=1
    )

    x_centers = 0.5 * (bins_x[1:] + bins_x[:-1])
    y_centers = 0.5 * (bins_y[2:] + bins_y[:-2])

    norm = Normalize(0, 1)
    plt.figure(figsize=(10, 6))
    img = plt.contourf(
        x_centers,
        y_centers,
        img_cdf.T,
        levels=quantiles,
        norm=norm,
        cmap="magma",
    )
    for i in range(0, 13, 1):
        l_q = plt.plot(x_val, y_pred[:, i], lw=2, ls="--", color="grey")
    handles = l_q
    handles += plt.plot(x_val, y_mean, c="k", ls="--", lw=2)
    labels = ["Predicted quantiles", "Predicted mean"]
    plt.legend(handles=handles, labels=labels)

    plt.xlim([-np.pi, np.pi])
    plt.ylim([-3, 3])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(False)
    plt.colorbar(img, label=r"Empirical quantiles")
