"""
==============
quatnn.metrics
==============

This module define metric object that can be used to evaluate mode
performance during training.
"""
from abc import ABC, abstractmethod, abstractproperty

from quantnn.backends import get_tensor_backend

class Metric(ABC):
    """
    The interface for metrics to compute during training.
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

    @abstractmethod
    def get_results(self):
        """
        Returns the values of the metric for all model outputs or
        only on
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
                dy = self.model.posterior_mean(y_pred=y_pred)
                cache["dy_mean"] = dy

        # Calculate the squared error.
        se = self.squared_error.get(key, 0.0)
        self.squared_error[key] = se + (cache["dy_mean"] ** 2).sum()
        n = self.n_samples.get(key, 0.0)
        self.n_samples[key] = n + xp.size(y_pred)

    def reset(self):
        self.squared_error = {}
        self.n_samples = {}

    def get_results(self):
        xp = self.tensor_backend
        if xp is None:
            return 0.0

        if len(self.squared_error) == 1:
            se = next(iter(self.squared_errors.items()))[1]
            n = next(iter(self.n_samples.items()))[1]
            return xp.to_numpy(se / n)[0]

        results = {}
        for k in self.squared_errors:
            results[k] = self.sqaured_errors[k] / self.n_samples[k]
        return results



