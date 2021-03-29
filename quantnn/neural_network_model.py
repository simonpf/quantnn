"""
============================
quantnn.neural_network_model
============================

This module defines the :py:class`NeuralNetworkModel` class, which
 provides and object-oriented interface to train, evaluate, store
and load neural network models.
"""

################################################################################
# Backend handling
################################################################################

#
# Try and load a supported backend.
#
import copy
import pickle
import importlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from quantnn.common import (QuantnnException,
                            UnsupportedBackendException,
                            ModelNotSupported)
from quantnn.logging import TrainingLogger

_DEFAULT_BACKEND = None

def set_default_backend(name):
    """
    Set the neural network package to use as backend.

    The currently available backend are "Keras" and "PyTorch".

    Args:
        name(str): The name of the backend.
    """
    global _DEFAULT_BACKEND
    if name.lower() == "keras":
        try:
            import quantnn.models.keras as keras
            _DEFAULT_BACKEND = keras
        except Exception as e:
            raise Exception("The following error occurred while trying "
                            " to import keras: ", e)
    elif name.lower() in ["pytorch", "torch"]:
        try:
            import quantnn.models.pytorch as pytorch
            _DEFAULT_BACKEND = pytorch
        except Exception as e:
            raise Exception("The following error occurred while trying "
                            " to import pytorch: ", e)
    else:
        raise Exception("\"{}\" is not a supported backend.".format(name))

def get_default_backend():

    global _DEFAULT_BACKEND

    if _DEFAULT_BACKEND is not None:
        return _DEFAULT_BACKEND

    try:
        import quantnn.models.pytorch as pytorch
        _DEFAULT_BACKEND = pytorch
        return _DEFAULT_BACKEND
    except ModuleNotFoundError:
        pytorch = None

    try:
        import quantnn.models.keras as keras
        _DEFAULT_BACKEND = keras
        return _DEFAULT_BACKEND
    except ModuleNotFoundError:
        keras = None

    if _DEFAULT_BACKEND is None:
        raise QuantnnException(
            "Couldn't load neither Keras nor PyTorch. You need to install "
            "at least one of those frameworks to use quantnn."
        )


def get_available_backends():
    backends = []
    try:
        import quantnn.models.pytorch as pytorch
        backends.append(pytorch)
    except ModuleNotFoundError:
        pass

    try:
        import quantnn.models.keras as keras
        backends.append(keras)
    except ModuleNotFoundError:
        pass
    return backends


class NeuralNetworkModel:
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 model):

        # Provided model is just an architecture tuple
        if type(model) == tuple:
            self.backend = get_default_backend()
            self.model = self.backend.FullyConnected(n_inputs,
                                                     n_outputs,
                                                     *model)
        # Provided model is predefined model.
        else:
            # Determine module and check if supported.
            self.model = None
            for backend in get_available_backends():
                try:
                    self.model = backend.Model.create(model)
                    self.backend = backend
                except ModelNotSupported:
                    pass
            if not self.model:
                raise UnsupportedBackendException(
                    "The provided model is not supported by any "
                    "of the backend modules.")

    def train(self,
              training_data,
              loss,
              validation_data=None,
              optimizer=None,
              scheduler=None,
              n_epochs=None,
              adversarial_training=None,
              device='cpu',
              logger=None,
              keys=None):
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
        return self.model.train(training_data,
                                validation_data=validation_data,
                                loss=loss,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                n_epochs=n_epochs,
                                adversarial_training=adversarial_training,
                                device=device,
                                logger=logger,
                                keys=keys)

    @staticmethod
    def load(path):
        r"""
        Load a model from a file.

        This loads a model that has been stored using the
        :py:meth:`quantnn.QRNN.save`  method.

        Arguments:

            path(str): The path from which to read the model.

        Return:

            The loaded QRNN object.
        """
        with open(path, 'rb') as file:
            qrnn = pickle.load(file)
            backend = importlib.import_module(qrnn.backend)
            qrnn.backend = backend
            model = backend.load_model(file)
            qrnn.model = backend.Model.create(model)
        return qrnn

    def save(self, path):
        r"""
        Store the QRNN model in a file.

        This stores the model to a file using pickle for all attributes that
        support pickling. The Keras model is handled separately, since it can
        not be pickled.

        Arguments:

            path(str): The path including filename indicating where to
                    store the model.

        """
        with open(path, "wb") as file:
            pickle.dump(self, file)

            old_class = self.model.__class__
            if self.model.__class__.__name__ == "__QuantnnMixin__":
                self.model.__class__ = self.model.__class__.__bases__[1]
            self.backend.save_model(file, self.model)
            self.model.__class__ = old_class

    def __getstate__(self):
        dct = copy.copy(self.__dict__)
        dct.pop("model")
        dct["backend"] = self.backend.__name__
        return dct

    def __setstate__(self, state):
        self.__dict__ = state
        self.model = None
