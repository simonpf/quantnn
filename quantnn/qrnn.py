r"""
============
quantnn.qrnn
============

This module provides the QRNN class, which is a generic implementation of
quantile regression neural networks (QRNN).

In essence, the QRNN class combines a backbone neural network with a set
of quantiles given by the corresponding quantile fractions
:math:`\{\tau_0, \ldots \tau_{n - 1}\}` with :math:`\tau_i \in [0, 1]` that
the network should learn to predict.

Example usage
-------------

.. code-block ::

    import numpy as np
    from quantnn.qrnn import QRNN
    from quantnn.models.pytorch import FullyConnected

    quantiles = np.linspace(0.01, 0.99, 99)
    # A fully-connected neural network with 10 inputs and one output for
    # each quantile.
    model = FullyConnected(n_inputs=10, n_outputs=99, n_layers=3, width=128)
    qrnn = QRNN(quantiles=quantiles, model=model)

    # Train the model
    x = np.random.rand(10, 10)
    y = np.random.rand(10, 1)
    qrnn.train(training_data=(x, y), n_epochs=1)

    # Perform inference.
    y_pred = qrnn.predict(x)

    # Save the model
    qrnn.save("qrnn.pckl")

    # Load the model
    QRNN.load("qrnn.pckl")
"""
import numpy as np
import quantnn.quantiles as qq
from quantnn.neural_network_model import NeuralNetworkModel
from quantnn.common import QuantnnException, UnsupportedBackendException
from quantnn.generic import softmax, to_array, get_array_module
from quantnn.utils import apply

###############################################################################
# QRNN class
###############################################################################
class QRNN(NeuralNetworkModel):
    r"""
    Quantile Regression Neural Network (QRNN)

    This class provides a generic implementation of  quantile regression
    neural networks, which can be used to estimate quantiles of the posterior
    distribution of remote sensing retrievals or any other probabilistic
    regression problem.

    The :class:`QRNN`` class wraps around a neural network rergession model,
    which may come from any of the supported ML backends, and adds functionality
    to train the neural network model as well as to perform inference on input
    data.

    Given a selection of quantile fractions :math:`{\tau_0},\ldots,y_{\tau_n} \in [0, 1]`,
    the network is trained to predict the corresponding quantiles :math:`y_{\tau_i}` of
    the posterior of the posterior distribution by training to minimize the sum of the
    loss functions

    .. math::

            \mathcal{L}_{\tau_i}(y_{\tau_i}, y_{true}) =
            \begin{cases} (1 - {\tau_i})|y_{\tau_i} - y_{true}| & \text{ if } y_{\tau_i} < y_\text{true} \\
            \tau_i |y_{\tau_i} - y_\text{true}| & \text{ otherwise, }\end{cases}

    where :math:`y_\text{true}` is the true value of the retrieval quantity.

    Attributes:

        quantiles (numpy.array):
            The 1D-array containing the quantiles :math:`\tau \in [0, 1]`
            that the network learns to predict.
        model:
            The neural network regression model used to predict the quantiles.
    """

    def __init__(
        self, quantiles, n_inputs=None, model=(3, 128, "relu"), transformation=None
    ):
        """
        Create a QRNN model.

        Arguments:
            n_inputs(int):
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
        self.n_inputs = n_inputs
        self.n_outputs = len(quantiles)
        self.quantiles = np.array(quantiles)
        super().__init__(self.n_inputs, self.n_outputs, model)
        self.quantile_axis = self.model.channel_axis
        self.transformation = transformation

    def train(
        self,
        training_data,
        validation_data=None,
        batch_size=None,
        optimizer=None,
        scheduler=None,
        n_epochs=None,
        adversarial_training=None,
        device="cpu",
        mask=None,
        logger=None,
        metrics=None,
        keys=None,
    ):
        """
        Train the underlying neural network model on given training data.

        The training data can be provided as either as tuples ``(x, y)``
        containing the raw input data as numpy arrays or as backend-specific
        dataset objects.

        .. note::

           If the train method doesn't serve your needs, the QRNN class can
           also be used with a pre-trained neural network.

        Args:
            training_data: Tuple of numpy arrays or backend-specific dataset
                object to use to train the model.
            validation_data: Optional validation data in the same format as the
                training data.
            batch_size: If training data is provided as arrays, this batch size
                will be used to for the training.
            optimizer: A backend-specific optimizer object to use for training.
            scheduler: A backend-specific scheduler object to use for training.
            n_epochs: The maximum number of epochs for which to train  the model.
            device: A ``str`` or backend-specific device object identifying the
                device to use for training.
            mask: Optional numeric value to use to mask all values that are
                smaller than or equal to this value.
            logger: A custom logger object to use to log training process. If
                not provided the default ``quantnn.training.TrainingLogger``
                class will be used.
            keys: Keys to use to determine input (``x``) and expected output
                 (``y``) when dataset elements are given as dictionaries.
        """
        loss = self.backend.QuantileLoss(self.quantiles, mask=mask)
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
            transformation=self.transformation,
        )

    def predict(self, x):
        r"""
        Predict quantiles of the conditional distribution :math:`p(y|x)``.

        Forward propagates the inputs in ``x`` through the network to
        obtain the predicted quantiles ``y_pred``.

        Arguments:

            x(np.array): Rank-k tensor containing the input data with
                the input channels (or features) for each sample located
                along its first dimension.

        Returns:

            Rank-k tensor ``y_pred`` containing the quantiles of each input
            sample along its first dimension
        """

        def transform(x, t):
            if t is None:
                return x
            return t.invert(x)

        if self.transformation is None:
            return self.model.predict(x)
        return apply(transform, self.model.predict(x), self.transformation)

    def cdf(self, x=None, y_pred=None, **kwargs):
        r"""
        Approximate the posterior CDF for given inputs ``x``.

        Propagates the inputs in ``x`` forward through the network and
        approximates the posterior CDF using a piecewise linear function.

        The piecewise linear function is given by its at quantiles
        :math:`y_{\tau_i}`` for :math:`\tau = \{0.0, \tau_1, \ldots,
        \tau_k, 1.0\}` where :math:`\tau_i` are the quantile fractions to be
        predicted by the network. The values for :math:`y_{\tau={0.0}}`
        and :math:`x_{\tau={1.0}}` are computed using

        .. math::

            y_{\tau=0.0} = 2.0 x_{\tau_1} - x_{\tau_2}

            y_{\tau=1.0} = 2.0 x_{\tau_k} - x_{\tau_{k-1}}

        Arguments:

            x: Rank-k tensor containing the input data with
                the input channels (or features) for each sample located
                along its first dimension.
            y_pred: Optional pre-computed quantile predictions, which, when
                 provided, will be used to avoid repeated propagation of the
                 the inputs through the network.

        Returns:

            Tuple ``(y_cdf, cdf)`` containing the abscissa-values ``y_cdf`` and
            the ordinates values ``cdf`` of the piece-wise linear approximation
            of the CDF :math:`F(y)`.

        """
        if y_pred is None:
            if x is None:
                raise ValueError(
                    "One of the input arguments x or y_pred must be " " provided."
                )
            y_pred = self.predict(x)

        def calculate_cdf(y_pred):
            module = get_array_module(y_pred)
            quantiles = to_array(module, self.quantiles, like=y_pred)
            return qq.cdf(y_pred, quantiles, quantile_axis=self.quantile_axis)

        return apply(calculate_cdf, y_pred)

    def pdf(self, x=None, y_pred=None, **kwargs):
        r"""
        Approximate the posterior probability density function (PDF) for given
        inputs ``x``.

        The PDF is approximated by computing the derivative of the piece-wise
        linear approximation of the CDF as computed by the
        :py:meth:`quantnn.QRNN.cdf` function.

        Arguments:

            x: Rank-k tensor containing the input data with
                the input channels (or features) for each sample located
                along its first dimension.
            y_pred: Optional pre-computed quantile predictions, which, when
                 provided, will be used to avoid repeated propagation of the
                 the inputs through the network.

        Returns:

            Tuple (x_pdf, y_pdf) containing the array with shape `(n, k)`  containing
            the x and y coordinates describing the PDF for the inputs in ``x``.

        """
        if y_pred is None:
            if x is None:
                raise ValueError(
                    "One of the input arguments x or y_pred must be " " provided."
                )
            y_pred = self.predict(x)

            def calculate_pdf(y_pred):
                module = get_array_module(y_pred)
                quantiles = to_array(module, self.quantiles, like=y_pred)
                return qq.pdf(y_pred, quantiles, quantile_axis=self.quantile_axis)

            return apply(calculate_pdf, y_pred)

    def sample_posterior(self, x=None, y_pred=None, n_samples=1, **kwargs):
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

        def calculate_samples(y_pred):
            module = get_array_module(y_pred)
            quantiles = to_array(module, self.quantiles, like=y_pred)
            return qq.sample_posterior(
                y_pred, quantiles, n_samples=n_samples, quantile_axis=self.quantile_axis
            )

        return apply(calculate_samples, y_pred)

    def sample_posterior_gaussian_fit(self, x=None, y_pred=None, n_samples=1):
        r"""
        Generates :code:`n` samples from the predicted posterior
        distribution for the input vector :code:`x`. The sampling
        is performed using a Gaussian fit to the predicted quantiles.

        Arguments:

            x: Rank-k tensor containing the input data with the input channels
                (or features) for each sample located along its first dimension.
            y_pred: Optional pre-computed quantile predictions, which, when
                 provided, will be used to avoid repeated propagation of the
                 the inputs through the network.
            n(int): The number of samples to generate.

        Returns:

            Tuple (xs, fs) containing the :math:`x`-values in `xs` and corresponding
            values of the posterior CDF :math:`F(x)` in `fs`.
        """
        if y_pred is None:
            if x is None:
                raise ValueError(
                    "One of the input arguments x or y_pred must be " " provided."
                )
            y_pred = self.predict(x)

        def calculate_samples(y_pred):
            module = get_array_module(y_pred)
            quantiles = to_array(module, self.quantiles, like=y_pred)
            return qq.sample_posterior_gaussian(
                y_pred, quantiles, n_samples=n_samples, quantile_axis=self.quantile_axis
            )

        return apply(calculate_samples, y_pred)

    def posterior_mean(self, x=None, y_pred=None, **kwargs):
        r"""
        Computes the posterior mean by computing the first moment of the
        predicted posterior CDF.

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

        def calculate_mean(y_pred):
            module = get_array_module(y_pred)
            quantiles = to_array(module, self.quantiles, like=y_pred)
            return qq.posterior_mean(
                y_pred, quantiles, quantile_axis=self.quantile_axis
            )

        return apply(calculate_mean, y_pred)

    def crps(self, x=None, y_pred=None, y_true=None, **kwargs):
        r"""
        Compute the Continuous Ranked Probability Score (CRPS).

        This function uses a piece-wise linear fit to the approximate posterior
        CDF obtained from the predicted quantiles in :code:`y_pred` to
        approximate the continuous ranked probability score (CRPS):

        .. math::
            \text{CRPS}(\mathbf{y}, x) = \int_{-\infty}^\infty (F_{x | \mathbf{y}}(x')
            - \mathrm{1}_{x < x'})^2 \: dx'

        Arguments:

            x: Rank-k tensor containing the input data with the input channels
                (or features) for each sample located along its first dimension.
            y_pred: Optional pre-computed quantile predictions, which, when
                 provided, will be used to avoid repeated propagation of the
                 the inputs through the network.
            y_true: Array containing the `n` true values, i.e. samples of the
                 true conditional distribution predicted by the QRNN.

            quantiles: 1D array containing the `k` quantile fractions :math:`\tau`
                       that correspond to the columns in `y_pred`.

        Returns:

            Tensor of rank k-1 containing the CRPS values for each of the samples.
        """
        if y_pred is None:
            if x is None:
                raise ValueError(
                    "One of the input arguments x or y_pred must be " " provided."
                )
            y_pred = self.predict(x)
        if y_true is None:
            raise ValueError(
                "The y_true argument must be provided to calculate "
                "the CRPS provided."
            )

        def calculate_crps(y_pred):
            module = get_array_module(y_pred)
            quantiles = to_array(module, self.quantiles, like=y_pred)
            return qq.crps(y_pred, y_true, quantiles, quantile_axis=self.quantile_axis)

        return apply(calculate_crps, y_pred)

    def probability_larger_than(self, x=None, y=None, y_pred=None, **kwargs):
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

        def calculate_prob(y_pred):
            module = get_array_module(y_pred)
            quantiles = to_array(module, self.quantiles, like=y_pred)
            return qq.probability_larger_than(
                y_pred, quantiles, y, quantile_axis=self.quantile_axis
            )

        return apply(calculate_prob, y_pred)

    def probability_less_than(self, x=None, y=None, y_pred=None, **kwargs):
        """
        Calculate probability of the output value being smaller than a
        given numeric threshold.

        Args:
            x: Rank-k tensor containing the input data with the input channels
                (or features) for each sample located along its first dimension.
            y_pred: Optional pre-computed quantile predictions, which, when
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

        def calculate_prob(y_pred):
            module = get_array_module(y_pred)
            quantiles = to_array(module, self.quantiles, like=y_pred)
            return qq.probability_less_than(
                y_pred, quantiles, y, quantile_axis=self.quantile_axis
            )

        return apply(calculate_prob, y_pred)

    def posterior_quantiles(self, x=None, y_pred=None, quantiles=None, **kwargs):
        r"""
        Compute the posterior quantiles.

        Arguments:

            x: Rank-k tensor containing the input data with the input channels
                (or features) for each sample located along its first dimension.
            y_pred: Optional pre-computed quantile predictions, which, when
                 provided, will be used to avoid repeated propagation of the
                 the inputs through the network.
            new_quantiles: List of quantile fraction values :math:`\tau_i \in [0, 1]`.
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

        def calculate_quantiles(y_pred):
            module = get_array_module(y_pred)
            new_quantiles = to_array(module, quantiles, like=y_pred)
            current_quantiles = to_array(module, self.quantiles, like=y_pred)
            return qq.posterior_quantiles(
                y_pred,
                quantiles=current_quantiles,
                new_quantiles=new_quantiles,
                quantile_axis=self.quantile_axis,
            )

        return apply(calculate_quantiles, y_pred)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "transformation"):
            self.transformation = None

    def lightning(self, mask, optimizer=None, scheduler=None, metrics=None, name=None):
        """
        Get Pytorch Lightning module.
        """
        from quantnn.models.pytorch.lightning import QuantnnLightning

        loss = self.backend.QuantileLoss(self.quantiles, mask=mask)
        return QuantnnLightning(
            self,
            loss,
            scheduler=scheduler,
            optimizer=optimizer,
            metrics=metrics,
            transformation=self.transformation,
            name=name,
            mask=mask
        )
