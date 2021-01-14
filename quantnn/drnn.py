"""
quantnn.drnn
============

This module provides a high-level implementation of a density regression
neural network, i.e. a network that predicts conditional probabilities
using a binned approximation of the probability density function.
"""
import numpy as np
import scipy
from scipy.special import softmax

from quantnn.common import QuantnnException
from quantnn.neural_network_model import NeuralNetworkModel

try:
    import quantnn.models.keras as keras
    default_backend = keras
except ModuleNotFoundError:
    pass

try:
    import quantnn.retrieval.qrnn.models.pytorch as pytorch
    default_backend = pytorch
except ModuleNotFoundError:
    pass

def _to_categorical(y, bins):
    """
    Converts scalar values to discrete, categorical representation where each
    value is represented by a bin index.

    Values that lie outside the provided range of

    Arguments:
        y: The values to discretize.
        bins: The n bins to use to discretize y represented by the n + 1
             corresponding, monotonuously increasing bin edges.

    Returns:
        Array of same shape as y containing the bin indices corresponding
        to each value.
    """
    return np.digitize(y, bins[:-1])

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
              device='cpu'):
        """
        Train model on given training data.

        The training is performed on the provided training data and an
        optionally-provided validation set. Training can use the following
        augmentation methods:
            - Gaussian noise added to input
            - Adversarial training
        The learning rate is decreased gradually when the validation or training
        loss did not decrease for a given number of epochs.

        Args:
            training_data: Tuple of numpy arrays or a dataset object to use to
                train the model.
            loss: Loss object to use as training criterion.
            validation_data: Optional validation data in the same format as the
                training data.
            batch_size: If training data is provided as arrays, this batch size
                will be used to for the training.
            optimizer: Optimizer object to use to train the model. Defaults to
                Adam optimizer.
            scheduler: Learning rate scheduler to use to schedule the learning
                rate. Defaults to plateau scheduler with a reduction factor
                of 10.0 and a patience of 5 epochs.
            n_epochs: The number of epochs for which to train the model.
            adversarial_training(``float`` or ``None``): Whether or not to
                perform adversarial training using the fast gradient sign
                method. When ``None`` no adversarial training is performed.
                When a ``float`` value is given this value will be used as
                the adversarial-training step length.
            device: "cpu" or "gpu" depending on whether the model should
                should be trained on CPU or GPU.

        Returns:
            Dictionary containing the training and validation losses.
        """
        if type(training_data) == tuple:
            x_train, y_train = training_data
            y_train = _to_categorical(y_train, self.bins[:-1])
            training_data = (x_train, y_train)
            if (validation_data):
                x_val, y_val = validation_data
                y_val = _to_categorical(y_val, self.bins[:-1])
                validation_data = x_val, y_val

        print(training_data[1].shape)

        loss = self.backend.CrossEntropyLoss()
        return self.model.train(training_data,
                                validation_data=validation_data,
                                loss=loss,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                n_epochs=n_epochs,
                                adversarial_training=adversarial_training,
                                device=device)

    def predict(self,
                x):
        y_pred = self.model.predict(x)
        y_pred = softmax(y_pred, axis=-1)
        x = 0.5 * (self.bins[1:] + self.bins[:-1])
        norm = np.tapz(y_pred, x=x, axis=-1, keepdims=True)
        return y_pred / norm
