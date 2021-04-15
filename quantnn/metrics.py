"""
==============
quatnn.metrics
==============

This module defines metric object that can be used to evaluate model
performance during training.
"""
from abc import ABC, abstractmethod, abstractproperty

import matplotlib.pyplot as plt
import numpy as np

from quantnn.backends import get_tensor_backend

class Metric(ABC):
    """
    The Metric abstract base class defines the basic interface for metric
    objects.
    """
    @abstractproperty
    def name(self):
        """
        The name to use to store the metric.
        """

    @abstractproperty
    def model(self):
        """
        Reference to the model instance for which the metrics are
        evaluated.
        """

    @abstractmethod
    def reset(self):
        """
        Called at the beginning of every epoch to singal
        to the metric to reset all accumulated values.
        """

    @abstractmethod
    def process_batch(self, key, y_pred, y, cache=None):
        """
        Calculate metrics for a batch of  model predictions and
        corresponding reference values for a specific model
        output.

        Args:
            key: Key identifying the model output.
            y_pred: The model prediction for the current batch.
            y: The reference values for the current batch.
            cache: Key-specific cache that can be used to share
                values between metrics.
        """

################################################################################
# Scalar metrics
################################################################################

class Bias(Metric):
    """
    Calculates the bias (mean error).
    """
    def __init__(self):
        self.error = {}
        self.n_samples = {}
        self._model = None
        self.tensor_backend = None

    @property
    def name(self):
        return "Bias"

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model


    def process_batch(self, key, y_pred, y, cache=None):
        if self.tensor_backend == None:
            self.tensor_backend = get_tensor_backend(y_pred)
        xp = self.tensor_backend

        # Get deviation from mean from cache
        # if not already computed.
        if cache is not None and "y_mean" in cache:
            y_mean = cache["y_mean"]
        else:
            y_mean = self.model.posterior_mean(y_pred=y_pred)
        if cache is not None:
            cache["y_mean"] = y_mean

        dy = y_mean - y

        # Calculate the error.
        e = self.error.get(key, 0.0)
        self.error[key] = e + dy.sum()
        n = self.n_samples.get(key, 0.0)
        self.n_samples[key] = n + xp.size(y_pred)

    def reset(self):
        self.error = {}
        self.n_samples = {}

    def get_values(self):
        xp = self.tensor_backend
        if xp is None:
            return 0.0

        if len(self.error) == 1:
            e = next(iter(self.error.items()))[1]
            n = next(iter(self.n_samples.items()))[1]
            return xp.to_numpy(e / n)[0]

        results = {}
        for k in self.error:
            results[k] = self.error[k] / self.n_samples[k]
        return results

class MeanSquaredError(Metric):
    """
    Mean squared error metric computed using the posterior mean.
    """
    def __init__(self):
        self.squared_error = {}
        self.n_samples = {}
        self._model = None
        self.tensor_backend = None

    @property
    def name(self):
        return "MSE"

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model


    def process_batch(self, key, y_pred, y, cache=None):
        if self.tensor_backend == None:
            self.tensor_backend = get_tensor_backend(y_pred)
        xp = self.tensor_backend

        # Get deviation from mean from cache
        # if not already computed.
        if cache is not None and "y_mean" in cache:
            y_mean = cache["y_mean"]
        else:
            y_mean = self.model.posterior_mean(y_pred=y_pred)
        if cache is not None:
            cache["y_mean"] = y_mean

        dy = y_mean - y

        # Calculate the squared error.
        se = self.squared_error.get(key, 0.0)
        self.squared_error[key] = se + (dy ** 2).sum()
        n = self.n_samples.get(key, 0.0)
        self.n_samples[key] = n + xp.size(y_pred)

    def reset(self):
        self.squared_error = {}
        self.n_samples = {}

    def get_values(self):
        xp = self.tensor_backend
        if xp is None:
            return 0.0

        if len(self.squared_error) == 1:
            se = next(iter(self.squared_error.items()))[1]
            n = next(iter(self.n_samples.items()))[1]
            return xp.to_numpy(se / n)[0]

        results = {}
        for k in self.squared_error:
            results[k] = self.squared_error[k] / self.n_samples[k]
        return results

################################################################################
# Calibration plot
################################################################################

class CalibrationPlot(Metric):
    """
    Produces a plot of  the calibration of the predicted quantiles of a QRNN.
    Currently only works in combination with the tensor board logger.
    """
    def __init__(self):
        self.calibration = {}
        self.n_samples = {}
        self._model = None
        self.tensor_backend = None
        self.mask = None

    @property
    def name(self):
        return "Calibration"

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):

        self._model = model

    def process_batch(self, key, y_pred, y, cache=None):

        if not hasattr(self._model, "quantiles"):
            return None
        quantiles = self._model.quantiles

        if self.tensor_backend == None:
            self.tensor_backend = get_tensor_backend(y_pred)
        xp = self.tensor_backend

        # Get deviation from mean from cache
        # if not already computed.
        axes = list(range(len(y.shape)))
        del axes[self._model.quantile_axis]

        if self.mask is not None:
            valid_pixels = xp.as_type(y > self.mask, y)
            valid_predictions = valid_pixels * xp.as_type(y <= y_pred, y_pred)

            c = self.calibration.get(key, xp.zeros(len(quantiles), y))
            self.calibration[key] = c + valid_predictions.sum(axes)
            n = self.calibration.get(key, xp.zeros(len(quantiles), y))
            self.n_samples[key] = n + valid_pixels.sum()
        else:
            valid_predictions = xp.as_type(y <= y_pred, y_pred)

            c = self.calibration.get(key, xp.zeros(len(quantiles), y))
            self.calibration[key] = c + valid_predictions.sum(axes)
            n = self.calibration.get(key, xp.zeros(len(quantiles), y))
            self.n_samples[key] = n + xp.size(y)

    def reset(self):
        self.calibration = {}
        self.n_samples = {}

    def make_calibration_plot(self, key):
        """
        Plots the calibration for a given target key using
        matplotlib.

        Args:
            key: Name of the target for which to plot the
                 calibration.

        Return:
            matplotlib Figure object containing the calibration plot.
        """
        if not hasattr(self._model, "quantiles"):
            return None

        xp = self.tensor_backend
        n_right = self.calibration[key]
        n_total = self.n_samples[key]

        cal = xp.to_numpy(n_right / n_total)
        qs = self.model.quantiles

        plt.ioff()
        f, ax = plt.subplots(1, 1, dpi=100)

        x = np.linspace(0, 1, 21)
        ax.plot(x, x, ls="--", c="k")
        ax.plot(qs, cal)

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel("Nominal frequency")
        ax.set_ylabel("Observed frequency")

        f.tight_layout()
        return f

    def get_figures(self):
        """
        Return:
             In the case of a single-target model directly returns the
             matplotlib figure containing the calibration plot. In the
             case of a multi-target model returns a dict of figures.
        """
        figures = {k: self.make_calibration_plot(k) for k in self.calibration}
        if len(figures) == 1:
            return next(iter(figures.items()))[1]
        return figures

################################################################################
# HeatMap
################################################################################

class ScatterPlot(Metric):
    """
    Produces a scatter plot of the posterior mean against the target value.
    """
    def __init__(self, bins=None, log_scale=False):
        self.bins = bins
        self.log_scale = log_scale
        self.n_samples = {}
        self._model = None
        self.tensor_backend = None
        self.mask = None

        self.y_pred = {}
        self.y = []

    @property
    def name(self):
        return "ScatterPlot"

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):

        self._model = model

    def process_batch(self, key, y_pred, y, cache=None):

        if self.tensor_backend == None:
            self.tensor_backend = get_tensor_backend(y_pred)
        xp = self.tensor_backend

        # Get deviation from mean from cache
        # if not already computed.
        if cache is not None and "y_mean" in cache:
            y_mean = cache["y_mean"]
        else:
            y_mean = self.model.posterior_mean(y_pred=y_pred)
        if cache is not None:
            cache["y_mean"] = y_mean

        y_mean = xp.to_numpy(y_mean)
        y = xp.to_numpy(y)

        if self.mask is not None:
            y_mean = y_mean[y.squeeze() > self.mask]

        self.y_pred.setdefault(key, []).append(y_mean.ravel())

        if len(self.y) < len(self.y_pred[key]):
            if self.mask is not None:
                y = y[y > self.mask]
            self.y.append(y)

    def reset(self):
        self.y_pred = {}
        self.y = []

    def make_scatter_plot(self, key):
        """
        Plots a 2D histogram of the predictions against the
        target values.

        Args:
            key: Name of the target for which to plot the
                 results.

        Return:
            matplotlib Figure object containing the scatter plot.
        """
        xp = self.tensor_backend

        y_pred = np.concatenate(self.y_pred[key])
        y = np.concatenate(self.y)
        img, x_edges, y_edges = np.histogram2d(y, y_pred, bins=self.bins)

        plt.ioff()
        f, ax = plt.subplots(1, 1, dpi=100)

        ax.pcolormesh(x_edges, y_edges, img.T)
        ax.plot(x_edges, y_edges, ls="--", c="grey")

        ax.set_xlim([x_edges[0], x_edges[-1]])
        ax.set_ylim([y_edges[0], y_edges[-1]])
        ax.set_xlabel("Target value")
        ax.set_ylabel("Posterior mean")

        if self.log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")

        f.tight_layout()
        return f

    def get_figures(self):
        """
        Return:
             In the case of a single-target model directly returns the
             matplotlib figure containing the calibration plot. In the
             case of a multi-target model returns a dict of figures.
        """
        figures = {k: self.make_scatter_plot(k) for k in self.y_pred}
        if len(figures) == 1:
            return next(iter(figures.items()))[1]
        return figures
