"""
quantnn.models.keras
====================

This module provides Keras neural network models that can be used as backend
models with the :py:class:`quantnn.QRNN` class.
"""
from collections.abc import Mapping
from copy import copy
import logging
import tempfile
import tarfile
import shutil
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Activation, deserialize
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import Loss, BinaryCrossentropy, SparseCategoricalCrossentropy

from quantnn.common import (
    QuantnnException,
    ModelNotSupported,
    InputDataError,
    DatasetError,
)
from quantnn.logging import TrainingLogger


def save_model(f, model):
    """
    Save keras model.

    Args:
        f(:code:`str` or binary stream): Either a path or a binary stream
            to store the data to.
        model(:code:`keras.models.Models`): The Keras model to save
    """
    path = tempfile.mkdtemp()
    filename = os.path.join(path, "keras_model")
    keras.models.save_model(model, filename)
    archive = tarfile.TarFile(fileobj=f, mode="w")
    archive.add(filename, arcname="keras_model")
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
    tar_file.extractall(path=path)
    filename = os.path.join(path, "keras_model")

    custom_objects = {
        "FullyConnected": FullyConnected,
        "QuantileLoss": QuantileLoss,
        "CrossEntropyLoss": CrossEntropyLoss,
    }
    model = keras.models.load_model(filename, custom_objects=custom_objects)
    shutil.rmtree(path)
    return model


###############################################################################
# Quantile loss
###############################################################################

LOGGER = logging.getLogger(__name__)


class CrossEntropyLoss(Loss):
    """
    Wrapper class around Keras' SparseCategoricalCrossEntropy class to support
    masking of input values.
    """

    def __init__(self, bins_or_classes, mask=None):
        name = "CrossEntropyLoss"
        super().__init__(
            reduction="none",
            name=name
        )
        self.__name__ = name
        self.bins = None
        self.n_classes = None
        if isinstance(bins_or_classes, int):
            self.n_classes = bins_or_classes
            if self.n_classes < 1:
                raise ValueError(
                    "The cross entropy loss is only meaningful for more than "
                    "1 class."
                )
            if bins_or_classes == 2:
                self.loss = BinaryCrossentropy(
                    from_logits=True,
                    name=name,
                    reduction="none"
                )
            else:
                self.loss = SparseCategoricalCrossentropy(
                    from_logits=True,
                    name=name,
                    reduction="none"
                )

        else:
            self.bins = [b for b in bins_or_classes]
            self.loss = SparseCategoricalCrossentropy(
                from_logits=True,
                name=name,
                reduction="none"
            )

        self.mask = mask


    def call(self, y_true, y_pred):
        if self.bins is not None:
            y_b = tf.raw_ops.Bucketize(input=y_true, boundaries=self.bins[1:-1], name=None)
        else:
            y_b = tf.math.maximum(y_true, 0)

        l = self.loss.call(y_b, y_pred)
        if self.mask is None:
            return tf.math.reduce_mean(l)
        mask = tf.cast(y_true > self.mask, tf.float32)
        l = tf.where(tf.squeeze(y_true > self.mask, -1), l, tf.zeros_like(l))
        return tf.math.reduce_sum(l) / tf.math.reduce_sum(mask)

    def get_config(self):
        return {"mask": self.mask}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __repr__(self):
        return f"CrossEntropyLoss(mask={self.mask})"


class QuantileLoss:
    """
    Wrapper class for the quantile error loss function. A class is used here
    to allow the implementation of a custom ``__repr`` function, so that the
    loss function object can be easily loaded using ``keras.model.load``.

    Attributes:

        quantiles: List of quantiles that should be estimated with
                   this loss function.

    """

    def __init__(self, quantiles, mask=None, quantile_axis=-1):
        self.__name__ = "QuantileLoss"
        self.quantiles = tf.convert_to_tensor(quantiles, dtype=tf.float32)
        self.mask = mask
        self.quantile_axis = quantile_axis

    def __call__(self, y_true, y_pred):
        n_quantiles = len(self.quantiles)

        quantile_shape = [1] * len(y_pred.shape)
        quantile_shape[self.quantile_axis] = -1
        quantiles = tf.reshape(self.quantiles, quantile_shape)

        if len(y_true.shape) < len(y_pred.shape):
            y_true = tf.expand_dims(y_true, self.quantile_axis)

        dy = y_pred - y_true
        l = tf.where(dy > 0, (1.0 - quantiles) * dy, -quantiles * dy)

        if self.mask is not None:
            mask = tf.cast(y_true > self.mask, tf.float32)
            return tf.math.reduce_sum(mask * l) / (
                tf.math.reduce_sum(mask) * n_quantiles
            )

        return tf.math.reduce_mean(l)

    def get_config(self):
        return {
            "quantiles": self.quantiles,
            "mask": self.mask,
            "quantile_axis": self.quantile_axis,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __repr__(self):
        return "QuantileLoss(" + repr(self.quantiles) + ")"


###############################################################################
# Keras data generators
###############################################################################


class BatchedDataset:
    """
    Keras data loader that batches a given dataset of numpy arrays.
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
            self.bs = 8

        self.indices = np.random.permutation(x.shape[0])

    def __iter__(self):
        self.i = 0
        return self

    def __len__(self):
        return self.x.shape[0] // self.bs

    def __next__(self):
        if self.i > len(self):
            raise StopIteration()

        inds = self.indices[
            np.arange(self.i * self.bs, (self.i + 1) * self.bs) % self.indices.size
        ]
        x_batch = np.copy(self.x[inds])
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

    def __init__(self, training_data, keys=None, sigma_noise=None):
        """
        Args:
            training_data: Data generator providing the original (noise-free)
                training data.
            sigma_noise: Vector the length of the input dimensions specifying
                the standard deviation of the noise.
        """
        self.training_data = training_data
        self.keys = keys
        self.sigma_noise = sigma_noise
        self.iterator = iter(training_data)

    def __iter__(self):
        return self

    def __len__(self):
        if hasattr(self.training_data, "__len__"):
            return len(self.training_data)
        else:
            return 1024

    def __next__(self):
        try:
            batch_data = next(self.iterator)
            if isinstance(batch_data, Mapping):
                if self.keys is not None:
                    try:
                        x_key, y_key = self.keys
                    except ValueError:
                        raise DatasetError(
                            f"Could not unpack provided keys f{self.keys} into "
                            "variables 'x_key, y_key'"
                        )
                else:
                    try:
                        x_key, y_key = batch_data.keys()
                    except ValueError as v:
                        raise DatasetError(
                            f"Could not unpack batch keys f{batch_data.keys()} into "
                            "variables 'x_key, y_key'"
                        )
                try:
                    x_batch = batch_data[x_key]
                    y_batch = batch_data[y_key]
                except Exception as e:
                    raise DatasetError(
                        "The following error was encountered when trying to "
                        f"retrieve the keys '{x_key}' and '{y_key} from a batch of  "
                        f"training data.: {e}"
                    )
            else:
                x_batch, y_batch = batch_data

        except StopIteration:
            self.iterator = iter(self.training_data)
            return next(self)
        if self.sigma_noise is not None:
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

    def __init__(self, validation_data, sigma_noise=None):
        self.validation_data = validation_data
        self.sigma_noise = sigma_noise
        self.iterator = iter(self.validation_data)

    def __iter__(self):
        return self

    # def __len__(self):
    #    if hasattr(self.validation_data, "__len__"):
    #        return len(self.validation_data)
    #    return 1

    def __next__(self):
        try:
            x_val, y_val = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.validation_data)
            raise StopIteration()

        if not self.sigma_noise is None:
            x_val += np.random.randn(*self.x_val.shape) * self.sigma_noise
        return (x_val, y_val)


###############################################################################
# LRDecay
###############################################################################


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


class CosineAnnealing(keras.callbacks.Callback):
    """
    Cosine annealing learning rate schedule.
    """

    def __init__(self, eta_max, eta_min, t_tot):
        """
        Args:
            eta_max: The maximum learning rate to use at t = 0
            eta_min: The minimum learning rate
            t_tot: The number of steps to go from eta_max to eta_min.
        """
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.t_tot = t_tot
        self.t = 0

    def on_train_begin(self, logs={}):
        self.t = 0

    def on_epoch_end(self, epoch, logs={}):
        lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (
            1.0 + np.cos(self.t / self.t_tot * np.pi)
        )
        keras.backend.set_value(self.model.optimizer.lr, lr)
        if lr < self.eta_min:
            self.model.stop_training = True
        self.t += 1


class LogCallback(keras.callbacks.Callback):
    """
    Adapter class to use generic quantnn logging interface with Keras model.
    """

    def __init__(self, logger, training_data, validation_data):
        """
        Create log callback.

        Args:
             training_data: The training data generator.
             validation_data: The validation data generator.
        """
        self.logger = logger
        super().__init__()

        # Number of training samples
        if hasattr(training_data, "__len__"):
            self.n_training_samples = len(training_data)
        else:
            self.n_training_samples = None

        # Number of validation samples
        if hasattr(validation_data, "__len__"):
            self.n_validation_samples = len(training_data)
        else:
            self.n_validation_samples = None

    def on_train_batch_end(self, batch, logs=None):
        """Log training batch end."""
        self.logger.training_step(logs["loss"], 1, of=self.n_training_samples)

    def on_test_batch_end(self, batch, logs=None):
        """Log validation batch end."""
        self.logger.validation_step(logs["loss"], 1, of=self.n_validation_samples)

    def on_epoch_begin(self, epoch, logs=None):
        """Log epoch beginning."""
        self.logger.epoch_begin(self.model)

    def on_epoch_end(self, epoch, logs=None):
        """Log epoch end."""
        lr = self.model.optimizer._decayed_lr("float32").numpy()
        self.logger.epoch(lr)


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
        super().__init__()

    @staticmethod
    def create(model):
        if not isinstance(model, keras.Model):
            raise ModelNotSupported(
                f"The provided model ({model}) is not supported by the Keras" "backend"
            )
        if not isinstance(model, KerasModel):
            model.__class__ = type("__QuantnnMixin__", (KerasModel, type(model)), {})
        return model

    @property
    def channel_axis(self):
        """
        The index of the axis that contains the channel information in a batch
        of input data.
        """
        format = keras.backend.image_data_format()
        if format.lower() == "channels_first":
            return 1
        return -1

    def reset(self):
        """
        Reinitialize the state of the model.
        """
        self.reset_states()

    def train(
        self,
        training_data,
        validation_data=None,
        loss=None,
        optimizer=None,
        scheduler=None,
        n_epochs=None,
        adversarial_training=None,
        batch_size=None,
        device="cpu",
        logger=None,
        metrics=None,
        keys=None,
        transformation=None,
    ):

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
        self.compile(loss=loss, optimizer=optimizer)

        #
        # Setup training generator
        #
        training_generator = TrainingGenerator(training_data, keys=keys)
        if adversarial_training:
            inputs = [self.input, self.targets[0], self.sample_weights[0]]
            input_gradients = tf.function(
                inputs, tf.gradients(self.total_loss, self.input)
            )
            training_generator = AdversarialTrainingGenerator(
                training_generator, input_gradients, adversarial_training
            )

        if validation_data is None:
            validation_generator = None
        else:
            validation_generator = ValidationGenerator(validation_data)

        if logger is None:
            logger = TrainingLogger(n_epochs)
        log = LogCallback(logger, training_generator, validation_generator)
        with tf.device(device):
            with logger:
                self.fit(
                    training_generator,
                    steps_per_epoch=len(training_generator),
                    epochs=n_epochs,
                    validation_data=validation_generator,
                    validation_steps=1,
                    callbacks=[scheduler, log],
                    verbose=False,
                )
        logger.training_end()

        def predict(self, *args, device="cpu", **kwargs):
            return keras.Model.predict(self, *args, **kwargs)


Model = KerasModel

###############################################################################
# Fully-connected network
###############################################################################


class FullyConnected(KerasModel, keras.Model):
    """
    A fully-connected network with a given depth and width.
    """

    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_layers,
        width,
        activation="ReLU",
        convolutional=False,
        **kwargs,
    ):
        """
        Create a fully-connected neural network.

        Args:
            n_inputs: The number of input features
            n_outputs: The number of output values
            n_layers: The number of layers in the network
            width: The number of neurons in each hidden layer.
            activation: The activation function to use for the hidden layers
                 given as name of a function in keras.activations or a Keras
                 Layer object.
            convolutional: If ``True``, the fully-connected network will be
                 constructed as a fully-connected network.
        """
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.width = width
        self.activation = activation
        self.convolutional = convolutional

        super().__init__(**kwargs)

        self.sequential = Sequential()

        n_in = n_inputs
        n_out = width

        if isinstance(activation, str):
            activation = layers.Activation(getattr(keras.activations, activation))

        if convolutional:
            for i in range(n_layers - 1):
                self.sequential.add(
                    layers.Conv2D(width, 1, input_shape=(None, None, n_in))
                )
                self.sequential.add(layers.BatchNormalization())
                self.sequential.add(activation)
                n_in = n_out
            self.sequential.add(layers.Conv2D(n_outputs, 1))
        else:
            for i in range(n_layers - 1):
                self.sequential.add(Dense(n_out, input_shape=(n_in,)))
                self.sequential.add(activation)
                n_in = n_out
            self.sequential.add(Dense(n_outputs, input_shape=(n_in,)))

        # Make sure model is built so that it can be saved.
        self.predict(np.zeros((10, n_inputs)))

    def call(self, input):
        return self.sequential.call(input)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "n_layers": self.n_layers,
            "activation": self.activation,
            "width": self.width,
            "convolutional": self.convolutional,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
