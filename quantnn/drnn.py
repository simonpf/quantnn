"""
quantnn.drnn
============

This module provides a high-level implementation of a density regression
neural network, i.e. a network that predicts conditional probabilities
using a binned approximation of the probability density function.
"""
import numpy as np
import scipy

import quantnn.density as qd
from quantnn.common import QuantnnException
from quantnn.generic import softmax, to_array, get_array_module
from quantnn.neural_network_model import NeuralNetworkModel
from quantnn.utils import apply


def _to_categorical(y, bins):
    """
    Converts scalar values to categorical representation where each value
    is represented by a bin index.

    Values that lie outside the provided range of

    Arguments:
        y: The values to discretize.
        bins: The n bins to use to discretize y represented by the n + 1
             corresponding, monotonuously increasing bin edges.

    Returns:
        Array of same shape as y containing the bin indices corresponding
        to each value.
    """
    return np.digitize(y, bins[1:-1])


class DRNN(NeuralNetworkModel):
    r"""
    Density regression neural network (DRNN).

    This class provider an high-level implementation of density regression
    neural networks aiming to provider a similar interface as the QRNN class.
    """

    def __init__(self, bins, n_inputs=None, model=(3, 128, "relu")):
        self.bins = bins
        if isinstance(self.bins, dict):
            n_bins = next(iter(self.bins.items()))[1].size
        else:
            n_bins = self.bins.size
        super().__init__(n_inputs, n_bins - 1, model)
        self.bin_axis = self.model.channel_axis

    def train(
        self,
        training_data,
        validation_data=None,
        optimizer=None,
        scheduler=None,
        n_epochs=None,
        adversarial_training=None,
        batch_size=None,
        device="cpu",
        mask=None,
        logger=None,
        metrics=None,
        keys=None,
    ):
        loss = self.backend.CrossEntropyLoss(self.bins, mask=mask)
        return super().train(
            training_data,
            loss,
            validation_data=validation_data,
            optimizer=optimizer,
            scheduler=scheduler,
            n_epochs=n_epochs,
            adversarial_training=adversarial_training,
            batch_size=batch_size,
            device=device,
            logger=logger,
            metrics=metrics,
            keys=keys,
        )

    def _post_process_prediction(self, y_pred, bins=None, key=None):
        module = get_array_module(y_pred)
        if bins is not None:
            bins = to_array(module, bins, like=y_pred)
        else:
            if isinstance(self.bins, dict):
                bins = to_array(module, self.bins[key], like=y_pred)
            else:
                bins = to_array(module, self.bins, like=y_pred)

        module = get_array_module(y_pred)
        y_pred = softmax(module, y_pred, axis=1)
        bins = to_array(module, bins, like=y_pred)
        y_pred = qd.normalize(y_pred, bins, bin_axis=self.bin_axis)
        return y_pred

    def predict(self, x):
        y_pred = self.model.predict(x)
        return apply(self._post_process_prediction, y_pred, self.bins)

    def posterior_mean(self, x=None, y_pred=None, key=None):
        r"""
        Computes the posterior mean by computing the first moment of the
        predicted posterior PDF.

        Arguments:

            x: Rank-k tensor containing the input data with the input channels
                (or features) for each sample located along its first dimension.
            y_pred: Optional pre-computed quantile predictions, which, when
                 provided, will be used to avoid repeated propagation of the
                 the inputs through the network.
        Returns:

            Tensor or rank k-1 the posterior means for all provided inputs.
        """
        if y_pred is None:
            if x is None:
                raise ValueError(
                    "One of the input arguments x or y_pred must be " " provided."
                )
            y_pred = self.predict(x)

        if key is None:
            bins = self.bins
        else:
            if isinstance(self.bins, dict):
                bins = self.bins[key]
            else:
                bins = self.bins

        def calculate_mean(y_pred, bins):
            module = get_array_module(y_pred)
            bins = to_array(module, bins, like=y_pred)
            return qd.posterior_mean(y_pred, bins, bin_axis=self.bin_axis)

        return apply(calculate_mean, y_pred, bins)

    def posterior_quantiles(self, x=None, y_pred=None, quantiles=None, key=None):
        r"""
        Compute the posterior quantiles.

        Arguments:

            x: Rank-k tensor containing the input data with the input channels
                (or features) for each sample located along its first dimension.
            y_pred: Optional pre-computed quantile predictions, which, when
                 provided, will be used to avoid repeated propagation of the
                 the inputs through the network.
            quantiles: List of quantile fraction values :math:`\tau_i \in [0, 1]`.
        Returns:

            Rank-k tensor containing the desired predicted quantiles along its
            first dimension.
        """
        if y_pred is None:
            if x is None:
                raise ValueError(
                    "One of the keyword arguments 'x' or 'y_pred'" " must be provided."
                )
            y_pred = self.predict(x)

        if quantiles is None:
            raise ValueError(
                "The 'quantiles' keyword argument must be provided to"
                "calculate the posterior quantiles."
            )

        if key is None:
            bins = self.bins
        else:
            if isinstance(self.bins, dict):
                bins = self.bins[key]
            else:
                bins = self.bins

        def calculate_quantiles(y_pred, bins):
            module = get_array_module(y_pred)
            bins = to_array(module, bins, like=y_pred)
            return qd.posterior_quantiles(
                y_pred, bins, quantiles, bin_axis=self.bin_axis
            )

        return apply(calculate_quantiles, y_pred, bins)

    def probability_larger_than(self, x=None, y=None, y_pred=None, key=None):
        """
        Calculate probability of the output value being larger than a
        given numeric threshold.

        Args:
            x: Rank-k tensor containing the input data with the input channels
                (or features) for each sample located along its first dimension.
            y: Optional pre-computed quantile predictions, which, when
                 provided, will be used to avoid repeated propagation of the
                 the inputs through the network.
            y: The threshold value.

        Returns:

            Tensor of rank k-1 containing the for each input sample the
            probability of the corresponding y-value to be larger than the
            given threshold.
        """
        if y_pred is None:
            if x is None:
                raise ValueError(
                    "One of the input arguments x or y_pred must be " " provided."
                )
            y_pred = self.predict(x)
        if y is None:
            raise ValueError(
                "The y argument must be provided to compute the " " probability."
            )

        if key is None:
            bins = self.bins
        else:
            if isinstance(self.bins, dict):
                bins = self.bins[key]
            else:
                bins = self.bins

        def calculate_probability(y_pred, bins):
            module = get_array_module(y_pred)
            bins = to_array(module, bins, like=y_pred)
            return qd.probability_larger_than(y_pred, bins, y, bin_axis=self.bin_axis)

        return apply(calculate_probability, y_pred, bins)

    def sample_posterior(self, x=None, y_pred=None, n_samples=1, key=None):
        r"""
        Generates :code:`n` samples from the predicted posterior distribution
        for the input vector :code:`x`. The sampling is performed by the
        inverse CDF method using the predicted CDF obtained from the
        :code:`cdf` member function.

        Arguments:

            x: Rank-k tensor containing the input data with
                the input channels (or features) for each sample located
                along its first dimension.
            y_pred: Optional pre-computed quantile predictions, which, when
                 provided, will be used to avoid repeated propagation of the
                 the inputs through the network.
            n: The number of samples to generate.

        Returns:

            Rank-k tensor containing the random samples for each input sample
            along the first dimension.
        """
        if y_pred is None:
            if x is None:
                raise ValueError(
                    "One of the input arguments x or y_pred must be " " provided."
                )
            y_pred = self.predict(x)

        if key is None:
            bins = self.bins
        else:
            if isinstance(self.bins, dict):
                bins = self.bins[key]
            else:
                bins = self.bins

        def calculate_samples(y_pred, bins):
            module = get_array_module(y_pred)
            bins = to_array(module, bins, like=y_pred)
            return qd.sample_posterior(
                y_pred, bins, n_samples=n_samples, bin_axis=self.bin_axis
            )

        return apply(calculate_samples, y_pred, bins)

    def quantile_function(self, x=None, y_pred=None, y=None, key=None):
        r"""
        Evaluate the quantile function a given y values.

        Arguments:

            x: Rank-k tensor containing the input data with
                the input channels (or features) for each sample located
                along its first dimension.
            y_pred: Optional pre-computed predicted pdf, which, when
                 provided, will be used to avoid repeated propagation of the
                 the inputs through the network.
            y: Rank-k tensor containing the values at which to evaluate the
                quantile function for each of the inputs in ``x``.

        Returns:

            Rank-k tensor containing the random samples for each input sample
            along the first dimension.
        """
        if y_pred is None:
            if x is None:
                raise ValueError(
                    "One of the input arguments x or y_pred must be " " provided."
                )
            y_pred = self.predict(x)

        if key is None:
            bins = self.bins
        else:
            if isinstance(self.bins, dict):
                bins = self.bins[key]
            else:
                bins = self.bins

        def calculate_quantile_function(y_pred, bins):
            module = get_array_module(y_pred)
            bins = to_array(module, bins, like=y_pred)
            return qd.quantile_function(y_pred, y, bins, bin_axis=self.bin_axis)

        return apply(calculate_quantile_function, y_pred, bins)

    def crps(self, x=None, y_pred=None, y_true=None, key=None):
        r"""
        Calculate CRPS score for given reference values.

        Arguments:

            x: Rank-k tensor containing the input data with
                the input channels (or features) for each sample located
                along its first dimension.
            y_pred: Optional pre-computed quantile predictions, which, when
                 provided, will be used to avoid repeated propagation of the
                 the inputs through the network.
            y_true: Rank-k tensor containing the true y values.

        Returns:

            Rank-k tensor containing crps values for all samples in x.
        """
        if y_pred is None:
            if x is None:
                raise ValueError(
                    "One of the input arguments x or y_pred must be "
                    " provided."
                )
            y_pred = self.predict(x)

        if key is None:
            bins = self.bins
        else:
            if isinstance(self.bins, dict):
                bins = self.bins[key]
            else:
                bins = self.bins

        def calculate_crps(y_pred, bins):
            module = get_array_module(y_pred)
            bins = to_array(module, bins, like=y_pred)
            return qd.crps(y_pred, y_true, bins, bin_axis=self.bin_axis)

        return apply(calculate_crps, y_pred, bins)
