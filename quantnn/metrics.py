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
from quantnn.common import InvalidDimensionException


def _check_input_dimensions(y_pred, y):
    """
    Ensures that input to a metric's 'process_batch' method have the same rank.

    Args:
         y_pred: The predictions given to the 'process_batch' method.
         y: The target values given to the 'process_batch' method.

    Raises:
         InvalidDimensionsException if ``y_pred`` and ``y`` don't match in
         rank.
    """
    if len(y.shape) != len(y_pred.shape):
        raise InvalidDimensionException(
            "The 'y' and 'y_pred' tensor arguments given to a metric must "
            "be of identical rank."
        )


def calculate_posterior_mean(model, y_pred, key, cache=None):
    """
    Lookup posterior mean from cache or try to calculate it.

    Args:
        model: The neural network model to use to calculate the posterior mean.
        y_pred: The post-processed predictions.
        key: The key of the output that is currently processed.
        cache: Optional cache to use to look up the posterior mean.

    Return:
        A ``torch.Tensor`` containing the posterior mean or None if the
        model doesn't provide this functionality.
    """
    try:
        if cache is not None and "y_mean" in cache:
            y_mean = cache["y_mean"]
        else:
            y_mean = model.posterior_mean(y_pred=y_pred, key=key)
            if cache is not None:
                cache["y_mean"] = y_mean
    except NotImplementedError:
        return None
    return y_mean


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


class ScalarMetric(Metric):
    def __init__(self):
        self._model = None
        self.keys = set()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @abstractmethod
    def get_value(self, key):
        """
        Return scalar value of the metric for give target.

        Args:
             key: The key identifying the target.
        """

    def get_values(self):
        """
        The values of the metric.

        Returns:

            In the case of a single-target model only a single scalar
            is returns.

            In the case of a multi-target model a dict is returned that
            maps the target keys to metric values.
        """
        if len(self.keys) == 1:
            return self.get_value(next(iter(self.keys)))

        results = {}
        for key in self.keys:
            try:
                results[key] = self.get_value(key)
            except KeyError:
                pass

        return results


class Bias(ScalarMetric):
    """
    Calculates the bias (mean error).
    """

    def __init__(self):
        super().__init__()
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
        _check_input_dimensions(y_pred, y)
        if hasattr(self.model, "_post_process_prediction"):
            y_pred = self.model._post_process_prediction(y_pred, key=key)

        y_mean = calculate_posterior_mean(self.model, y_pred, key, cache=cache)
        if y_mean is None:
            return None

        self.keys.add(key)

        if self.tensor_backend is None:
            self.tensor_backend = get_tensor_backend(y_pred)
        xp = self.tensor_backend


        if len(y.shape) > len(y_mean.shape):
            if hasattr(self.model, "quantile_axis"):
                dist_axis = self.model.quantile_axis
            else:
                dist_axis = self.model.bin_axis
            y = y.squeeze(dist_axis)
        dy = y_mean - y

        # Calculate the error.
        if self.mask is not None:
            mask = xp.as_type(y > self.mask, y)
            e = self.error.get(key, 0.0)
            self.error[key] = e + (mask * dy).sum()
            n = self.n_samples.get(key, 0.0)
            self.n_samples[key] = n + mask.sum()
        else:
            e = self.error.get(key, 0.0)
            self.error[key] = e + dy.sum()
            n = self.n_samples.get(key, 0.0)
            self.n_samples[key] = n + xp.size(y)

    def reset(self):
        self.error = {}
        self.n_samples = {}

    def get_value(self, key):
        xp = self.tensor_backend
        if xp is None:
            return 0.0
        return xp.to_numpy(self.error[key] / self.n_samples[key])


class MeanSquaredError(ScalarMetric):
    """
    Mean squared error metric computed using the posterior mean.
    """

    def __init__(self):
        super().__init__()
        self.squared_error = {}
        self.n_samples = {}
        self.tensor_backend = None

    @property
    def name(self):
        return "MSE"

    def process_batch(self, key, y_pred, y, cache=None):
        _check_input_dimensions(y_pred, y)
        if hasattr(self.model, "_post_process_prediction"):
            y_pred = self.model._post_process_prediction(y_pred, key=key)

        self.keys.add(key)

        if self.tensor_backend is None:
            self.tensor_backend = get_tensor_backend(y_pred)
        xp = self.tensor_backend

        y_mean = calculate_posterior_mean(self.model, y_pred, key, cache=cache)
        if y_mean is None:
            return None

        if len(y.shape) > len(y_mean.shape):
            if hasattr(self.model, "quantile_axis"):
                dist_axis = self.model.quantile_axis
            else:
                dist_axis = self.model.bin_axis
            y = y.squeeze(dist_axis)

        dy = y_mean - y

        # Calculate the squared error.
        if self.mask is not None:
            mask = xp.as_type(y > self.mask, y)
            se = self.squared_error.get(key, 0.0)
            self.squared_error[key] = se + ((mask * dy) ** 2).sum()
            n = self.n_samples.get(key, 0.0)
            self.n_samples[key] = n + mask.sum()
        else:
            se = self.squared_error.get(key, 0.0)
            self.squared_error[key] = se + (dy ** 2).sum()
            n = self.n_samples.get(key, 0.0)
            self.n_samples[key] = n + xp.size(y)

    def reset(self):
        self.squared_error = {}
        self.n_samples = {}

    def get_value(self, key):
        xp = self.tensor_backend
        if xp is None:
            return 0.0

        return xp.to_numpy(self.squared_error[key] / self.n_samples[key])


class Correlation(ScalarMetric):
    """
    Correlation coefficent computed using the posterior mean.
    """
    def __init__(self):
        super().__init__()

        self.xy = {}
        self.xx = {}
        self.yy = {}
        self.x = {}
        self.y = {}

        self.n_samples = {}
        self.tensor_backend = None

    @property
    def name(self):
        return "Correlation"

    def process_batch(self, key, y_pred, y, cache=None):
        _check_input_dimensions(y_pred, y)
        if hasattr(self.model, "_post_process_prediction"):
            y_pred = self.model._post_process_prediction(y_pred, key=key)

        self.keys.add(key)

        if self.tensor_backend is None:
            self.tensor_backend = get_tensor_backend(y_pred)
        xp = self.tensor_backend

        y_mean = calculate_posterior_mean(self.model, y_pred, key, cache=cache)
        if y_mean is None:
            return None

        if len(y.shape) > len(y_mean.shape):
            if hasattr(self.model, "quantile_axis"):
                dist_axis = self.model.quantile_axis
            else:
                dist_axis = self.model.bin_axis
            y = y.squeeze(dist_axis)

        if self.mask is not None:
            mask = xp.as_type(y > self.mask, y)
            y_pred_m = mask * y_mean
            y_m = mask * y
            n = self.n_samples.get(key, 0.0)
            self.n_samples[key] = n + xp.to_numpy(mask.sum())
        else:
            y_pred_m = y_mean
            y_m = y
            n = self.n_samples.get(key, 0.0)
            self.n_samples[key] = n + xp.size(y_m)

        x = self.x.get(key, 0.0)
        self.x[key] = x + xp.to_numpy((y_pred_m).sum())
        xx = self.xx.get(key, 0.0)
        self.xx[key] = xx + xp.to_numpy((y_pred_m ** 2).sum())

        y = self.y.get(key, 0.0)
        self.y[key] = y + xp.to_numpy((y_m).sum())
        yy = self.yy.get(key, 0.0)
        self.yy[key] = yy + xp.to_numpy((y_m ** 2).sum())

        xy = self.xy.get(key, 0.0)
        self.xy[key] = xy + xp.to_numpy((y_pred_m * y_m).sum())

    def reset(self):
        self.x = {}
        self.xx = {}
        self.y = {}
        self.yy = {}
        self.xy = {}
        self.n_samples = {}

    def get_value(self, key):

        xp = self.tensor_backend
        if xp is None:
            return 0.0

        xy = self.xy[key]
        x = self.x[key]
        xx = self.xx[key]
        y = self.y[key]
        yy = self.yy[key]
        n = self.n_samples[key]

        sigma_x = np.sqrt(xx / n - (x / n) ** 2)
        sigma_y = np.sqrt(yy / n - (y / n) ** 2)
        corr = (xy / n - x / n * y / n) / sigma_x / sigma_y

        return corr


class CRPS(ScalarMetric):
    """
    Mean squared error metric computed using the posterior mean.
    """

    def __init__(self):
        super().__init__()
        self.crps = {}
        self.tensor_backend = None

    @property
    def name(self):
        return "CRPS"

    def process_batch(self, key, y_pred, y, cache=None):
        _check_input_dimensions(y_pred, y)
        if hasattr(self.model, "_post_process_prediction"):
            y_pred = self.model._post_process_prediction(y_pred, key=key)

        self.keys.add(key)

        if self.tensor_backend is None:
            self.tensor_backend = get_tensor_backend(y_pred)
        xp = self.tensor_backend

        crps = self.model.crps(y_pred=y_pred, y_true=y, key=key)
        if crps is None:
            return None

        crps_batches = self.crps.setdefault(key, [])
        crps = xp.to_numpy(crps)
        y = xp.to_numpy(y)

        if self.mask is not None:
            if hasattr(self.model, "quantile_axis"):
                dist_axis = self.model.quantile_axis
            else:
                dist_axis = self.model.bin_axis
            if len(y.shape) > len(crps.shape):
                y = y.squeeze(dist_axis)

            crps = crps[y > self.mask]

        crps_batches.append(crps.ravel())

    def reset(self):
        self.crps = {}

    def get_value(self, key):
        xp = self.tensor_backend
        if xp is None:
            return 0.0

        crps = np.concatenate(self.crps[key])
        return crps.mean()


################################################################################
# Calibration plot
################################################################################


class CalibrationPlot(Metric):
    """
    Produces a plot of  the calibration of the predicted quantiles of a QRNN.
    Currently only works in combination with the tensor board logger.
    """

    def __init__(self, quantiles=None):
        self.calibration = {}
        self.n_samples = {}
        self._model = None
        self.tensor_backend = None
        self.mask = None
        self.quantiles = None

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
        _check_input_dimensions(y_pred, y)
        if hasattr(self.model, "_post_process_prediction"):
            y_pred = self.model._post_process_prediction(y_pred, key=key)

        if hasattr(self._model, "quantiles"):
            quantiles = self._model.quantiles
        else:
            quantiles = self.quantiles
            if quantiles is None:
                quantiles = np.linspace(0.05, 0.95, 10)
            y_pred = self.model.posterior_quantiles(
                y_pred=y_pred, quantiles=quantiles, key=key
            )
            if y_pred is None:
                return None

        if self.tensor_backend is None:
            self.tensor_backend = get_tensor_backend(y_pred)
        xp = self.tensor_backend

        # Get deviation from mean from cache
        # if not already computed.
        axes = list(range(len(y.shape)))
        if hasattr(self.model, "quantile_axis"):
            q_axis = self.model.quantile_axis
        else:
            q_axis = self.model.bin_axis

        del axes[q_axis]

        if self.mask is not None:
            valid_pixels = xp.as_type(y > self.mask, y)
            valid_predictions = valid_pixels * xp.as_type(y <= y_pred, y_pred)

            c = self.calibration.get(key, xp.zeros(len(quantiles), y))
            self.calibration[key] = c + valid_predictions.sum(axes)
            n = self.n_samples.get(key, xp.zeros(len(quantiles), y))
            self.n_samples[key] = n + valid_pixels.sum()
        else:
            valid_predictions = xp.as_type(y <= y_pred, y_pred)

            c = self.calibration.get(key, xp.zeros(len(quantiles), y))
            self.calibration[key] = c + valid_predictions.sum(axes)
            n = self.n_samples.get(key, xp.zeros(len(quantiles), y))
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
        if hasattr(self._model, "quantiles"):
            quantiles = self._model.quantiles
        else:
            quantiles = self.quantiles
            if quantiles is None:
                quantiles = np.linspace(0.05, 0.95, 10)

        xp = self.tensor_backend
        n_right = self.calibration[key]
        n_total = self.n_samples[key]

        cal = xp.to_numpy(n_right / n_total)

        plt.ioff()
        f, ax = plt.subplots(1, 1, dpi=100)

        x = np.linspace(0, 1, 21)
        ax.plot(x, x, ls="--", c="k")
        ax.plot(quantiles, cal)

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
        self.y = {}

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
        _check_input_dimensions(y_pred, y)
        if hasattr(self.model, "_post_process_prediction"):
            y_pred = self.model._post_process_prediction(y_pred, key=key)

        if self.tensor_backend is None:
            self.tensor_backend = get_tensor_backend(y_pred)
        xp = self.tensor_backend

        y_mean = calculate_posterior_mean(self.model, y_pred, key, cache=cache)
        if y_mean is None:
            return None

        y_mean = xp.to_numpy(y_mean)
        y = xp.to_numpy(y)

        if self.mask is not None:
            if hasattr(self.model, "quantile_axis"):
                dist_axis = self.model.quantile_axis
            else:
                dist_axis = self.model.bin_axis
            if len(y.shape) > len(y_mean.shape):
                y = y.squeeze(dist_axis)
            y_mean = y_mean[y > self.mask]
            y = y[y > self.mask]

        self.y_pred.setdefault(key, []).append(y_mean.ravel())
        self.y.setdefault(key, []).append(y.ravel())

    def reset(self):
        self.y_pred = {}
        self.y = {}

    def make_scatter_plot(self, key, add_title=False):
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
        y = np.concatenate(self.y[key])

        if isinstance(self.bins, dict):
            bins = dict[key]
        else:
            bins = self.bins
        if bins is None:
            y_min = min(y.min(), y.min())
            y_max = min(y.max(), y.max())
            if self.log_scale:
                y_max = np.log10(y_max)
                if y_min <= 0:
                    y_min = y_max - 4
                else:
                    y_min = np.log10(y_min)
                bins = np.logspace(y_min, y_max, 41)
            else:
                bins = np.linspace(y_min, y_max, 41)

        img, x_edges, y_edges = np.histogram2d(y, y_pred, bins=bins)
        norm = img.sum(axis=-1, keepdims=True)
        img /= norm

        plt.ioff()
        f, ax = plt.subplots(1, 1, dpi=100)

        m = ax.pcolormesh(x_edges, y_edges, img.T)
        ax.plot(x_edges, y_edges, ls="--", c="grey")
        plt.colorbar(m, label="Counts")
        if add_title:
            ax.set_title(key)

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
        add_title = False
        if len(self.y_pred) > 1:
            add_title = True
        figures = {k: self.make_scatter_plot(k, add_title) for k in self.y_pred}
        if len(figures) == 1:
            return next(iter(figures.items()))[1]
        return figures


class QuantileFunction(Metric):
    """
    Produces a plot of the distribution of the quantile function for true predictions.
    """

    def __init__(self):
        self.qfs = {}
        self._model = None
        self.tensor_backend = None
        self.mask = None

    @property
    def name(self):
        return "Quantile function"

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def process_batch(self, key, y_pred, y, cache=None):
        _check_input_dimensions(y_pred, y)
        if hasattr(self.model, "_post_process_prediction"):
            y_pred = self.model._post_process_prediction(y_pred, key=key)

        qf = self.model.quantile_function(y_pred=y_pred, y=y, key=key)

        if self.tensor_backend is None:
            self.tensor_backend = get_tensor_backend(y_pred)
        xp = self.tensor_backend

        qf = xp.to_numpy(qf)
        if self.mask is not None:
            y = xp.to_numpy(y)
            if hasattr(self.model, "quantile_axis"):
                dist_axis = self.model.quantile_axis
            else:
                dist_axis = self.model.bin_axis
            if len(y.shape) > len(qf.shape):
                y = y.squeeze(dist_axis)
            self.qfs.setdefault(key, []).append(qf[y > self.mask])
        else:
            self.qfs.setdefault(key, []).append(qf.ravel())

    def reset(self):
        self.qfs = {}

    def make_quantile_function_plot(self, key):
        """
        Plots the calibration for a given target key using
        matplotlib.

        Args:
            key: Name of the target for which to plot the
                 calibration.

        Return:
            matplotlib Figure object containing the calibration plot.
        """
        bins = np.linspace(0, 1, 41)
        qfs = np.concatenate(self.qfs[key])
        y, _ = np.histogram(self.qfs[key], bins=bins, density=True)

        plt.ioff()
        f, ax = plt.subplots(1, 1, dpi=100)

        x = 0.5 * (bins[1:] + bins[:-1])
        ax.plot(x, y)
        # ax.plot(x, 1.0 * np.ones(x.size), ls="--", c="k")

        # ax.set_xlim([0, 1])
        # ax.set_ylim([0, 2])
        ax.set_xlabel("F^{-1}(y_true)")
        ax.set_ylabel("p(F(y_true))")

        f.tight_layout()
        return f

    def get_figures(self):
        """
        Return:
             In the case of a single-target model directly returns the
             matplotlib figure containing the calibration plot. In the
             case of a multi-target model returns a dict of figures.
        """
        figures = {k: self.make_quantile_function_plot(k) for k in self.qfs}
        if len(figures) == 1:
            return next(iter(figures.items()))[1]
        return figures
