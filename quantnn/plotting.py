"""
quantnn.plotting
================

The plotting module provides some utility function for plotting QRNN results.
"""

from matplotlib.colors import to_rgba

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
