"""
==============
quatnn.metrics
==============

This module define metric object that can be used to evaluate mode
performance during training.
"""
from abc import ABC, abstractmethod, abstractproperty

import matplotlib.pyplot as plt
import numpy as np

from quantnn.backends import get_tensor_backend

class Metric(ABC):
    """
    The interface for metrics to compute during training.
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
        if cache is not None:
            if "dy" in cache:
                dy = cache["dy_mean"]
            else:
                dy = self.model.posterior_mean(y_pred=y_pred) - y
                cache["dy_mean"] = dy

        # Calculate the squared error.
        se = self.squared_error.get(key, 0.0)
        self.squared_error[key] = se + (cache["dy_mean"] ** 2).sum()
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
            se = next(iter(self.squared_errors.items()))[1]
            n = next(iter(self.n_samples.items()))[1]
            return xp.to_numpy(se / n)[0]

        results = {}
        for k in self.squared_error:
            results[k] = self.squared_error[k] / self.n_samples[k]
        return results


class Calibration(Metric):
    """
    The calibration of predicted quantiles.
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
        figures = {k: self.make_calibration_plot(k) for k in self.calibration}
        if len(figures) == 1:
            return next(iter(figures.items()))[1]
        return figures
