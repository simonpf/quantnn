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
from quantnn.common import QuantnnException, UnsupportedBackendException

_DEFAULT_BACKEND = None
try:
    import quantnn.models.keras as keras
    _DEFAULT_BACKEND = keras
except ModuleNotFoundError:
    pass
try:
    import quantnn.models.pytorch as pytorch
    _DEFAULT_BACKEND = pytorch
except ModuleNotFoundError:
    pass

if _DEFAULT_BACKEND == None:
    print(_DEFAULT_BACKEND)
    raise QuantnnException(
        "Couldn't load neither Keras nor PyTorch. You need to install "
        "at least one of those frameworks to use quantnn."
    )

def set_default_backend(name):
    """
    Set the neural network package to use as backend.

    The currently available backend are "Keras" and "PyTorch".

    Args:
        name(str): The name of the backend.
    """
    global backend
    if name.lower() == "keras":
        try:
            import quantnn.models.keras as keras
            backend = keras
        except Exception as e:
            raise Exception("The following error occurred while trying "
                            " to import keras: ", e)
    elif name.lower() in ["pytorch", "torch"]:
        try:
            import quantnn.models.pytorch as pytorch
            backend = pytorch
        except Exception as e:
            raise Exception("The following error occurred while trying "
                            " to import pytorch: ", e)
    else:
        raise Exception("\"{}\" is not a supported backend.".format(name))

class NeuralNetworkModel:
    def __init__(self,
                 input_dimensions,
                 output_dimensions,
                 model):

        # Provided model is just an architecture tuple
        if type(model) == tuple:
            self.backend = _DEFAULT_BACKEND
            model = self.backend.FullyConnected(input_dimensions,
                                                output_dimensions,
                                                model)
        # Provided model is predefined model.
        else:
            # Determine module and check if supported.
            module = model.__module__.split(".")[0]
            if module not in ["keras", "torch"]:
                raise UnsupportedBackendException(
                    "The provided model comes from a unsupported "
                    "backend module. ")
            self.backend = globals()[module]

            model = self.backend.Model.create(input_dimensions,
                                              output_dimensions,
                                              model)
        self.model = model

    def train(self,
              training_data,
              loss,
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
        return self.model.train(training_data,
                                validation_data=validation_data,
                                loss=loss,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                n_epochs=n_epochs,
                                adversarial_training=adversarial_training,
                                batch_size=batch_size,
                                device=device)
