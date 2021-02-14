"""
quantnn.qrnn
============

This module provides the QRNN class, which implements the high-level
functionality of quantile regression neural networks, while the neural
network implementation is left to the model backends implemented in the
``quantnn.models`` submodule.
"""
import copy
import pickle
import importlib

import numpy as np
import quantnn.quantiles as qq
from quantnn.neural_network_model import NeuralNetworkModel
from quantnn.common import QuantnnException, UnsupportedBackendException

################################################################################
# Set the backend
################################################################################


###############################################################################
# QRNN class
###############################################################################
class QRNN(NeuralNetworkModel):
    r"""
    Quantile Regression Neural Network (QRNN)

    This class provides a high-level implementation of  quantile regression
    neural networks. It can be used to estimate quantiles of the posterior
    distribution of remote sensing retrievals.

    The :class:`QRNN`` class uses an arbitrary neural network model, that is
    trained to minimize the quantile loss function

    .. math::
            \mathcal{L}_\tau(y_\tau, y_{true}) =
            \begin{cases} (1 - \tau)|y_\tau - y_{true}| & \text{ if } y_\tau < y_\text{true} \\
            \tau |y_\tau - y_\text{true}| & \text{ otherwise, }\end{cases}

    where :math:`x_\text{true}` is the true value of the retrieval quantity
    and and :math:`x_\tau` is the predicted quantile. The neural network
    has one output neuron for each quantile to estimate.

    The QRNN class provides a generic QRNN implementation in the sense that it
    does not assume a fixed neural network architecture or implementation.
    Instead, this functionality is off-loaded to a model object, which can be
    an arbitrary regression network such as a fully-connected or a
    convolutional network. A range of different models are provided in the
    quantnn.models module. The :class:`QRNN`` class just
    implements high-level operation on the QRNN output while training and
    prediction are delegated to the model object. For details on the respective
    implementation refer to the documentation of the corresponding model class.

    .. note::

      For the QRNN implementation :math:`x` is used to denote the input
      vector and :math:`y` to denote the output. While this is opposed
      to inverse problem notation typically used for retrievals, it is
      in line with machine learning notation and felt more natural for
      the implementation. If this annoys you, I am sorry.

    Attributes:
        backend(``str``):
            The name of the backend used for the neural network model.
        quantiles (numpy.array):
            The 1D-array containing the quantiles :math:`\tau \in [0, 1]`
            that the network learns to predict.
        model:
            The neural network regression model used to predict the quantiles.
    """
    def __init__(self,
                 quantiles,
                 input_dimensions=None,
                 model=(3, 128, "relu")):
        """
        Create a QRNN model.

        Arguments:
            input_dimensions(int):
                The dimension of the measurement space, i.e. the
                number of elements in a single measurement vector y
            quantiles(np.array):
                1D-array containing the quantiles  to estimate of
                the posterior distribution. Given as fractions within the range
                [0, 1].
            model:
                A (possibly trained) model instance or a tuple ``(d, w, act)``
                describing the architecture of a fully-connected neural network
                with :code:`d` hidden layers with :code:`w` neurons and
                :code:`act` activation functions.
        """
        self.input_dimensions = input_dimensions
        self.output_dimensions = len(quantiles)
        self.quantiles = np.array(quantiles)
        super().__init__(self.input_dimensions,
                         self.output_dimensions,
                         model)

    def train(self,
              training_data,
              validation_data=None,
              batch_size=None,
              optimizer=None,
              scheduler=None,
              n_epochs=None,
              adversarial_training=None,
              device='cpu',
              mask=None):
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
            training_data: Tuple of numpy arrays of a dataset object to use to
                train the model.
            validation_data: Optional validation data in the same format as the
                training data.
            batch_size: If training data is provided as arrays, this batch size
                will be used to for the training.
            sigma_noise: If training data is provided as arrays, training data
                will be augmented by adding noise with the given standard
                deviations to each input vector before it is presented to the
                model.
            adversarial_training(``bool``): Whether or not to perform
                adversarial training using the fast gradient sign method.
            delta_at: The scaling factor to apply for adversarial training.
            initial_learning_rate(``float``): The learning rate with which the
                 training is started.
            momentum(``float``): The momentum to use for training.
            convergence_epochs(``int``): The number of epochs with
                 non-decreasing loss before the learning rate is decreased
            learning_rate_decay(``float``): The factor by which the learning rate
                 is decreased.
            learning_rate_minimum(``float``): The learning rate at which the
                 training is aborted.
            maximum_epochs(``int``): For how many epochs to keep training.
            training_split(``float``): If no validation data is provided, this
                 is the fraction of training data that is used for validation.
            gpu(``bool``): Whether or not to try to run the training on the GPU.
        """
        loss = self.backend.QuantileLoss(self.quantiles, mask=mask)
        return self.model.train(training_data,
                                validation_data=validation_data,
                                loss=loss,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                n_epochs=n_epochs,
                                adversarial_training=adversarial_training,
                                batch_size=batch_size,
                                device=device)

    def predict(self, x):
        r"""
        Predict quantiles of the conditional distribution P(y|x).

        Forward propagates the inputs in `x` through the network to
        obtain the predicted quantiles `y`.

        Arguments:

            x(np.array): Array of shape `(n, m)` containing `n` m-dimensional inputs
                         for which to predict the conditional quantiles.

        Returns:

             Array of shape `(n, k)` with the columns corresponding to the k
             quantiles of the network.

        """
        return self.model.predict(x)

    def cdf(self, x):
        r"""
        Approximate the posterior CDF for given inputs `x`.

        Propagates the inputs in `x` forward through the network and
        approximates the posterior CDF by a piecewise linear function.

        The piecewise linear function is given by its values at approximate
        quantiles :math:`x_\tau`` for :math:`\tau = \{0.0, \tau_1, \ldots,
        \tau_k, 1.0\}` where :math:`\tau_k` are the quantiles to be estimated
        by the network. The values for :math:`x_{0.0}` and :math:`x_{1.0}` are
        computed using

        .. math::

            x_{0.0} = 2.0 x_{\tau_1} - x_{\tau_2}

            x_{1.0} = 2.0 x_{\tau_k} - x_{\tau_{k-1}}

        Arguments:

            x(np.array): Array of shape `(n, m)` containing `n` inputs for which
                         to predict the conditional quantiles.

        Returns:

            Tuple (xs, fs) containing the :math:`x`-values in `xs` and corresponding
            values of the posterior CDF :math:`F(x)` in `fs`.

        """
        y_pred = self.predict(x)
        return qq.cdf(y_pred, self.quantiles, quantile_axis=1)

    def calibration(self, *args, **kwargs):
        """
        Compute calibration curve for the given dataset.
        """
        return self.model.calibration(*args, *kwargs)

    def pdf(self, x):
        r"""
        Approximate the posterior probability density function (PDF) for given
        inputs ``x``.

        The PDF is approximated by computing the derivative of the piece-wise
        linear approximation of the CDF as computed by the
        :py:meth:`quantnn.QRNN.cdf` function.

        Arguments:

            x(np.array): Array of shape `(n, m)` containing `n` inputs for which
               to predict PDFs.

        Returns:

            Tuple (x_pdf, y_pdf) containing the array with shape `(n, k)`  containing
            the x and y coordinates describing the PDF for the inputs in ``x``.

        """
        y_pred = self.predict(x)
        return qq.pdf(y_pred, self.quantiles, quantile_axis=1)

    def sample_posterior(self, x, n_samples=1):
        r"""
        Generates :code:`n` samples from the estimated posterior
        distribution for the input vector :code:`x`. The sampling
        is performed by the inverse CDF method using the estimated
        CDF obtained from the :code:`cdf` member function.

        Arguments:

            x(np.array): Array of shape `(n, m)` containing `n` inputs for which
                         to predict the conditional quantiles.

            n(int): The number of samples to generate.

        Returns:

            Tuple (xs, fs) containing the :math:`x`-values in `xs` and corresponding
            values of the posterior CDF :math: `F(x)` in `fs`.
        """
        y_pred = self.predict(x)
        return qq.sample_posterior(y_pred,
                                   self.quantiles,
                                   n_samples=n_samples,
                                   quantile_axis=1)

    def sample_posterior_gaussian_fit(self, x, n_samples=1):
        r"""
        Generates :code:`n` samples from the estimated posterior
        distribution for the input vector :code:`x`. The sampling
        is performed by the inverse CDF method using the estimated
        CDF obtained from the :code:`cdf` member function.

        Arguments:

            x(np.array): Array of shape `(n, m)` containing `n` inputs for which
                         to predict the conditional quantiles.

            n(int): The number of samples to generate.

        Returns:

            Tuple (xs, fs) containing the :math:`x`-values in `xs` and corresponding
            values of the posterior CDF :math: `F(x)` in `fs`.
        """
        y_pred = self.predict(x)
        return qq.sample_posterior_gaussian(y_pred,
                                            self.quantiles,
                                            n_samples=n_samples,
                                            quantile_axis=1)

    def posterior_mean(self, x):
        r"""
        Computes the posterior mean by computing the first moment of the
        estimated posterior CDF.

        Arguments:

            x(np.array): Array of shape `(n, m)` containing `n` inputs for which
                         to predict the posterior mean.
        Returns:

            Array containing the posterior means for the provided inputs.
        """
        y_pred = self.predict(x)
        return qq.posterior_mean(y_pred,
                                 self.quantiles,
                                 quantile_axis=1)

    def crps(y_pred, y_true, quantiles):
        r"""
        Compute the Continuous Ranked Probability Score (CRPS) for given quantile
        predictions.

        This function uses a piece-wise linear fit to the approximate posterior
        CDF obtained from the predicted quantiles in :code:`y_pred` to
        approximate the continuous ranked probability score (CRPS):

        .. math::
            CRPS(\mathbf{y}, x) = \int_{-\infty}^\infty (F_{x | \mathbf{y}}(x')
            - \mathrm{1}_{x < x'})^2 \: dx'

        Arguments:

            y_pred(numpy.array): Array of shape `(n, k)` containing the `k`
                                 estimated quantiles for each of the `n`
                                 predictions.

            y_test(numpy.array): Array containing the `n` true values, i.e.
                                 samples of the true conditional distribution
                                 estimated by the QRNN.

            quantiles: 1D array containing the `k` quantile fractions :math:`\tau`
                       that correspond to the columns in `y_pred`.

        Returns:

            `n`-element array containing the CRPS values for each of the
            predictions in `y_pred`.
        """
        y_pred = self.predict(x)
        return qq.crps(y_pred,
                       self.quantiles,
                       y_true,
                       quantile_axis=1)

    def probability_larger_than(self, x, y):
        """
        Classify output based on posterior PDF and given numeric threshold.

        Args:
            x: The input data as :code:`np.ndarray` or backend-specific
               dataset object.
            threshold: The numeric threshold to apply for classification.
        """
        y_pred = self.predict(x)
        return qq.probability_larger_than(y_pred,
                                          self.quantiles,
                                          y,
                                          quantile_axis=1)


    def probability_less_than(self, x, y):
        """
        Classify output based on posterior PDF and given numeric threshold.

        Args:
            x: The input data as :code:`np.ndarray` or backend-specific
               dataset object.
            threshold: The numeric threshold to apply for classification.
        """
        y_pred = self.predict(x)
        return qq.probability_less_than(y_pred,
                                        self.quantiles,
                                        y,
                                        quantile_axis=1)
