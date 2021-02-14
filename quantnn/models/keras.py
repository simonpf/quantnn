"""
quantnn.models.keras
====================

This module provides Keras neural network models that can be used as backend
models with the :py:class:`quantnn.QRNN` class.
"""
import logging
import tempfile
import tarfile
import shutil
import os

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, deserialize
from keras.optimizers import SGD
from keras.losses import CategoricalCrossentropy as CrossEntropyLoss
import keras.backend as K

from quantnn.common import QuantnnException, ModelNotSupported

def save_model(f, model):
    """
    Save keras model.

    Args:
        f(:code:`str` or binary stream): Either a path or a binary stream
            to store the data to.
        model(:code:`keras.models.Models`): The Keras model to save
    """
    path = tempfile.mkdtemp()
    filename = os.path.join(path, "keras_model.h5")
    keras.models.save_model(model, filename)
    archive = tarfile.TarFile(fileobj=f, mode="w")
    archive.add(filename, arcname="keras_model.h5")
    archive.close()
    shutil.rmtree(path)

def load_model(file):
    """
    Load keras model.

    Args:
        file(:code:`str` or binary stream): Either a path or a binary stream
            to read the model from

    Returns:
        The loaded keras model.
    """
    path = tempfile.mkdtemp()
    tar_file = tarfile.TarFile(fileobj=file, mode="r")
    tar_file.extract("keras_model.h5", path=path)
    filename = os.path.join(path, "keras_model.h5")

    custom_objects = {
        "FullyConnected": FullyConnected,
        "QuantileLoss": QuantileLoss,
    }
    model = keras.models.load_model(filename, custom_objects=custom_objects)
    shutil.rmtree(path)
    return model


################################################################################
# Quantile loss
################################################################################

LOGGER = logging.getLogger(__name__)


def skewed_absolute_error(y_true, y_pred, tau):
    """
    The quantile loss function for a given quantile tau:

    L(y_true, y_pred) = (tau - I(y_pred < y_true)) * (y_pred - y_true)

    Where I is the indicator function.
    """
    dy = y_pred - y_true
    return K.mean((1.0 - tau) * K.relu(dy) + tau * K.relu(-dy), axis=-1)


def quantile_loss(y_true, y_pred, taus):
    """
    The quantiles loss for a list of quantiles. Sums up the error contribution
    from the each of the quantile loss functions.
    """
    e = skewed_absolute_error(K.flatten(y_true), K.flatten(y_pred[:, 0]), taus[0])
    for i, tau in enumerate(taus[1:]):
        e += skewed_absolute_error(K.flatten(y_true), K.flatten(y_pred[:, i + 1]), tau)
    return e


class QuantileLoss:
    """
    Wrapper class for the quantile error loss function. A class is used here
    to allow the implementation of a custom `__repr` function, so that the
    loss function object can be easily loaded using `keras.model.load`.

    Attributes:

        quantiles: List of quantiles that should be estimated with
                   this loss function.

    """

    def __init__(self, quantiles, mask=None):
        self.__name__ = "QuantileLoss"
        self.quantiles = quantiles
        self.maks = None

    def __call__(self, y_true, y_pred):
        return quantile_loss(y_true, y_pred, self.quantiles)

    def __repr__(self):
        return "QuantileLoss(" + repr(self.quantiles) + ")"

################################################################################
# Keras data generators
################################################################################


class BatchedDataset:
    """
    Keras data loader that batches a given dataset of numpy arryas.
    """

    def __init__(self, training_data, batch_size=None):
        """
        Create batched dataset.

        Args:
            training_data: Tuple :code:`(x, y)` containing the input
               and output data as arrays.
            batch_size(:code:`int`): The batch size
        """
        x, y = training_data
        self.x = x
        self.y = y

        if batch_size:
            self.bs = batch_size
        else:
            self.bs = 256

        self.indices = np.random.permutation(x.shape[0])
        self.i = 0

    def __iter__(self):
        LOGGER.info("iter...")
        return self

    def __len__(self):
        return self.x.shape[0] // self.bs

    def __next__(self):
        inds = self.indices[
            np.arange(self.i * self.bs, (self.i + 1) * self.bs) % self.indices.size
        ]
        x_batch = np.copy(self.x[inds, :])
        y_batch = self.y[inds]
        self.i = self.i + 1
        # Shuffle training set after each epoch.
        if self.i % (self.x.shape[0] // self.bs) == 0:
            self.indices = np.random.permutation(self.x.shape[0])

        return (x_batch, y_batch)


class TrainingGenerator:
    """
    This Keras sample generator takes a generator for noise-free training data
    and adds independent Gaussian noise to each of the components of the input.

    Attributes:
        training_data: Data generator providing the data
        sigma_noise: A vector containing the standard deviation of each
                 component.
    """

    def __init__(self,
                 training_data,
                 sigma_noise=None):
        """
        Args:
            training_data: Data generator providing the original (noise-free)
                training data.
            sigma_noise: Vector the length of the input dimensions specifying
                the standard deviation of the noise.
        """
        self.training_data = training_data
        self.sigma_noise = sigma_noise

    def __iter__(self):
        LOGGER.info("iter...")
        return self

    def __len__(self):
        return len(self.training_data)

    def __next__(self):
        x_batch, y_batch = next(self.training_data)
        if not self.sigma_noise is None:
            x_batch += np.random.randn(*x_batch.shape) * self.sigma_noise
        return (x_batch, y_batch)


class AdversarialTrainingGenerator:
    """
    This Keras sample generator takes the noise-free training data
    and adds independent Gaussian noise to each of the components
    of the input.

    Attributes:
        training_data: Training generator to use to generate the input
            data
        input_gradients: Keras function to compute the gradients of the
            network
        eps: The perturbation factor.
    """

    def __init__(self, training_data, input_gradients, eps):
        """
        Args:
            training_data: Training generator to use to generate the input
                data
            input_gradients: Keras function to compute the gradients of the
                network
            eps: The perturbation factor.
        """
        self.training_data = training_data
        self.input_gradients = input_gradients
        self.eps = eps

    def __iter__(self):
        LOGGER.info("iter...")
        return self

    def __len__(self):
        return len(self.training_data)

    def __next__(self):
        if self.i % 2 == 0:
            x_batch, y_batch = next(self.training_data)
            self.x_batch = x_batch
            self.y_batch = y_batch
        else:
            x_batch = self.x_batch
            y_batch = self.y_batch
            grads = self.input_gradients([x_batch, y_batch, 1.0])
            x_batch += self.eps * np.sign(grads)

        self.i = self.i + 1
        return x_batch, y_batch


class ValidationGenerator:
    """
    This Keras sample generator is similar to the training generator
    only that it returns the whole validation set and doesn't perform
    any randomization.

    Attributes:

        x_val: The validation input, i.e. the brightness temperatures
                 measured by the satellite.
        y_val: The validation output, i.e. the value of the retrieval
                 quantity.
        x_mean: A vector containing the mean of each input component.
        x_sigma: A vector containing the standard deviation of each
                 component.
    """

    def __init__(self, validation_data, sigma_noise):
        self.validation_data = validation_data
        self.sigma_noise = sigma_noise

    def __iter__(self):
        return self

    def __next__(self):
        x_val, y_val = next(self.validation_data)
        if not self.sigma_noise is None:
            x_val += np.random.randn(*self.x_val.shape) * self.sigma_noise
        return (x_val, self.y_val)


################################################################################
# LRDecay
################################################################################


class LRDecay(keras.callbacks.Callback):
    """
    The LRDecay class implements the Keras callback interface and reduces
    the learning rate according to validation loss reduction.

    Attributes:

        lr_decay: The factor c > 1.0 by which the learning rate is
                  reduced.
        lr_minimum: The training is stopped when this learning rate
                    is reached.
        convergence_steps: The number of epochs without validation loss
                           reduction required to reduce the learning rate.

    """

    def __init__(self, model, lr_decay, lr_minimum, convergence_steps):
        self.model = model
        self.lr_decay = lr_decay
        self.lr_minimum = lr_minimum
        self.convergence_steps = convergence_steps
        self.steps = 0

    def on_train_begin(self, logs={}):
        self.losses = []
        self.steps = 0
        self.min_loss = 1e30

    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get("val_loss")
        if loss is None:
            loss = logs.get("loss")
        self.losses += [loss]
        if not self.losses[-1] < self.min_loss:
            self.steps = self.steps + 1
        else:
            self.steps = 0
        if self.steps > self.convergence_steps:
            lr = keras.backend.get_value(self.model.optimizer.lr)
            keras.backend.set_value(self.model.optimizer.lr, lr / self.lr_decay)
            self.steps = 0
            LOGGER.info("\n Reduced learning rate to " + str(lr))

            if lr < self.lr_minimum:
                self.model.stop_training = True

        self.min_loss = min(self.min_loss, self.losses[-1])

################################################################################
# Default scheduler and optimizer
################################################################################

def _get_default_optimizer(schedule):
    """
    The default optimizer. Currently set to Adam optimizer.
    """
    return keras.optimizers.RMSprop()

def _get_default_scheduler(model):
    """
    The default optimizer. Currently set to Adam optimizer.
    """
    return LRDecay(model, 10.0, 1e-4, 5)


################################################################################
# QRNN
################################################################################

class KerasModel:
    r"""
    Base class for Keras models.

    This base class provides generic utility function for the training, saving
    and evaluation of Keras models.

    Attributes:
        input_dimensions (int): The input dimension of the neural network, i.e.
            the dimension of the measurement vector.
        quantiles (numpy.array): The 1D-array containing the quantiles
            :math:`\tau \in [0, 1]` that the network learns to predict.

        depth (int):
            The number layers in the network excluding the input layer.

        width (int):
            The width of the hidden layers in the network.

        activation (str):
            The name of the activation functions to use in the hidden layers
            of the network.

        models (list of keras.models.Sequential):
            The ensemble of Keras neural networks used for the quantile regression
            neural network.
    """
    def __init__(self, *args, **kwargs):
        """
        Forwards call to super to support multiple inheritance.
        """
        super().__init__(args, kwargs)

    @staticmethod
    def create(model):
        if not isinstance(model, keras.Model):
            raise ModelNotSupported(
                f"The provided model ({model}) is not supported by the Keras"
                "backend")

        if isinstance(model, KerasModel):
            return model
        new_model = KerasModel(input_dimension, quantiles)
        new_model.__bases__ = (model,)
        return new_model

    def reset(self):
        """
        Reinitialize the state of the model.
        """
        self.reset_states()

    def train(self,
              training_data,
              validation_data=None,
              loss=None,
              optimizer=None,
              scheduler=None,
              n_epochs=None,
              adversarial_training=None,
              batch_size=None,
              device='cpu'):

        # Input data.
        if type(training_data) == tuple:
            if not type(training_data[0]) == np.ndarray:
                raise QuantnnException(
                    "When training data is provided as tuple"
                    " (x, y) it must contain numpy arrays."
                )
            training_data = BatchedDataset(training_data, batch_size)

        if type(validation_data) is tuple:
            validation_data = BatchedDataset(validation_data, batch_size)

        # Compile model
        self.custom_objects = {loss.__name__: loss}
        if not scheduler:
            scheduler = _get_default_scheduler(self)
        if not optimizer:
            optimizer = _get_default_optimizer(scheduler)
        self.compile(loss=loss,
                     optimizer=optimizer)

        #
        # Setup training generator
        #
        training_generator = TrainingGenerator(training_data)
        if adversarial_training:
            inputs = [self.input, self.targets[0], self.sample_weights[0]]
            input_gradients = K.function(
                inputs, K.gradients(self.total_loss, self.input)
            )
            training_generator = AdversarialTrainingGenerator(
                training_generator, input_gradients, adversarial_training
            )

        if validation_data is None:
            validation_generator = None
        else:
            validation_generator = ValidationGenerator(validation_data, sigma_noise)

        self.fit(
            training_generator,
            steps_per_epoch=len(training_generator),
            epochs=n_epochs,
            validation_data=validation_generator,
            validation_steps=1,
            callbacks=[scheduler])


Model = KerasModel

###############################################################################
# Fully-connected network
###############################################################################


class FullyConnected(KerasModel, Sequential):
    """
    Keras implementation of fully-connected networks.
    """

    def __init__(self,
                 n_inputs=None,
                 n_outputs=None,
                 n_layers=None,
                 width=None,
                 activation=None,
                 **kwargs):
        """
        Create a fully-connected neural network.

        Args:
            input_dimension(:code:`int`): Number of input features
            quantiles(:code:`array`): The quantiles to predict given
                as fractions within [0, 1].
            n_layers: The number of hidden layers in the network.
            width: The number of neurons in the hidden layers.
            activation: The activation function to use after each linear
                 layers.
        """
        inputs = [n_inputs, n_outputs, n_layers, width, activation]
        if all([x is None for x in inputs]):
            super().__init__(**kwargs)
            return

        n_in = n_inputs
        n_out = width

        layers = []
        for i in range(n_layers - 1):
            layers.append(Dense(n_out, activation=activation, input_shape=(n_in,)))
            n_in = n_out

        layers.append(Dense(n_outputs, input_shape=(n_in,)))
        super().__init__(*layers)
