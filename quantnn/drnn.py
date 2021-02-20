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
from quantnn.generic import softmax
from quantnn.neural_network_model import NeuralNetworkModel

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
    def __init__(self,
                 bins,
                 input_dimensions=None,
                 model=(3, 128, "relu")):
        self.bins = bins
        super().__init__(input_dimensions, bins.size - 1, model)


    def train(self,
              training_data,
              validation_data=None,
              optimizer=None,
              scheduler=None,
              n_epochs=None,
              adversarial_training=None,
              device='cpu',
              mask=None):
        if type(training_data) == tuple:
            x_train, y_train = training_data
            y_train = _to_categorical(y_train, self.bins[:-1])
            training_data = (x_train, y_train)
            if (validation_data):
                x_val, y_val = validation_data
                y_val = _to_categorical(y_val, self.bins[:-1])
                validation_data = x_val, y_val

        loss = self.backend.CrossEntropyLoss(mask=mask)
        return self.model.train(training_data,
                                validation_data=validation_data,
                                loss=loss,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                n_epochs=n_epochs,
                                adversarial_training=adversarial_training,
                                device=device)

    def predict(self, x):
        y_pred = self.model.predict(x)
        y_pred = softmax(y_pred, axis=-1)
        norm = np.sum(y_pred * (self.bins[1:] - self.bins[:-1]), axis=-1, keepdims=True)
        return y_pred / norm

    def posterior_mean(self, x=None, y_pred=None):
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
                raise ValueError("One of the input arguments x or y_pred must be "
                                 " provided.")
            y_pred = self.predict(x)
        return qd.posterior_mean(y_pred,
                                 self.bins,
                                 quantile_axis=1)

    def posterior_quantiles(self, x=None, y_pred=None, quantiles=None):
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
                raise ValueError("One of the keyword arguments 'x' or 'y_pred'"
                                 " must be provided.")
            y_pred = self.predict(x)

        if quantiles is None:
            raise ValueError("The 'quantiles' keyword argument must be provided to"
                             "calculate the posterior quantiles.")

        return qd.posterior_quantiles(y_pred,
                                      self.bins,
                                      quantiles,
                                      quantile_axis=1)
