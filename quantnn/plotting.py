"""
================
quantnn.plotting
================

The plotting module provides some utility function for plotting QRNN results.
"""
from copy import copy
import pathlib

from matplotlib import rc
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

_STYLE_FILE = pathlib.Path(__file__).parent / "data" / "matplotlib_style.rc"


def set_style(latex=False):
    """
    Sets matplotlib style to a style file that I find visually more pleasing
    then the default settings.

    Args:
        latex: Whether or not to use latex to render text.
    """
    plt.style.use(str(_STYLE_FILE))
    rc("text", usetex=latex)


def plot_confidence_intervals(ax, x, y_pred, quantiles, color="C0"):
    """
    Plots symmetric confidence intervals using transparency to display uncertainty.

    This function plots a 1-dimensional sequence of predicts quantiles as confidence
    intervals. The intervals are displayed as filled regions with the transparency
    set according to the corresponding uncertainty.

    Arguments:
        ax: Matplotlib axes instance to use for plotting.
        x: The x values corresponding to the prediction in y_pred.
        y_pred: 2D array containing the predicted quantiles.
        quantiles: The quantiles corresponding to the second axis of y_pred.
        color: The color to use for filling.
    """
    n = y_pred.shape[1]
    if n % 2:
        c_0 = y_pred[:, n // 2]
    else:
        c_0 = 0.5 * (y_pred[:, n // 2] + y_pred[:, n // 2 + 1])
    alpha_0 = 0.9
    alpha_min = 0.1
    d_alpha = alpha_0 - alpha_min

    c = c_0
    for i in range(n // 2 - 1, -1, -1):
        q = quantiles[i]
        alpha = alpha_0 - 2.0 * d_alpha * (0.5 - q)
        color = to_rgba(color, alpha)
        ax.fill_between(x, c, y_pred[:, i], edgecolor=None, facecolor=color)

    c = c_0
    for i in range(n // 2 + 1, n):
        q = quantiles[i]
        alpha = alpha_0 - 2.0 * d_alpha * (q - 0.5)
        color = to_rgba(color, alpha)
        ax.fill_between(x, c, y_pred[:, i], edgecolor=None, facecolor=color)


def plot_quantiles(ax, x, y_pred, quantiles, cmap="magma"):
    """
    Plots symmetric confidence intervals using transparency to display uncertainty.

    This function plots a 1-dimensional sequence of predicts quantiles as confidence
    intervals. The intervals are displayed as filled regions with the transparency
    set according to the corresponding uncertainty.

    Arguments:
        ax: Matplotlib axes instance to use for plotting.
        x: The x values corresponding to the prediction in y_pred.
        y_pred: 2D array containing the predicted quantiles.
        quantiles: The quantiles corresponding to the second axis of y_pred.
        color: The color to use for filling.
    """
    cmap = copy(mpl.cm.get_cmap(cmap))
    norm = mpl.colors.BoundaryNorm(quantiles, cmap.N)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cmap.set_under([0.0] * 4)
    cmap.set_over([0.0] * 4)

    n = len(quantiles)
    for i in range(n - 1):
        q = 0.5 * (quantiles[i] + quantiles[i + 1])
        color = mappable.to_rgba(q)
        y_low = y_pred[:, i]
        y_high = y_pred[:, i + 1]
        ax.fill_between(x, y_low, y_high, edgecolor=None, facecolor=color)

    return mappable
