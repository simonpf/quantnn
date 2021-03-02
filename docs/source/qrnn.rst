===========================================
Quantile regression neural networks (QRNNs)
===========================================

Consider first the case of a simple, one-dimensional input vector :math:`x` with
input features :math:`x_1, \ldots, x_n` and a corresponding scalar output value
:math:`y`. If the problem of mapping :math:`x` to :math:`y` does not admit a unique
solution a more suitable approach than predicting a single value for :math:`y`
is to instead predict its conditional distribution :math:`P(y|x)` for given
input :math:`x`.

QRNNs do this by learning to predict a sequence of quantiles :math:`y_\tau` of
the distribution :math:`P(y | \mathbf{x})`. The quantile :math:`y_\tau` for
:math:`\tau \in [0, 1]` is defined as the value for which :math:`P(y \leq y_\tau
| x) = \tau`. Since the quantiles :math:`y_{\tau_0}, y_{\tau_1}, \ldots`
correspond to the values of the inverse :math:`F^{-1}` of the cumulative
distribution function :math:`F(x) = P(X \leq x | Y = y)`, the QRNN output can be
interpreted as a piece-wise linear approximation of the CDF of :math:`P(y|x)`.

.. figure:: qrnn.svg
  :width: 600
  :alt: Illustration of a QRNN

  A QRNN is a neural network which predicts a sequence of quantiles of
  the posterior distribution :math:`P(y|x)`.

How it works
------------

QRNNs make use of quantile regression to learn to predict the quantiles of the
distribution :math:`P(y|x)`. This works because the quantile :math:`y_\tau` for
a given quantile fraction :math:`\tau \in [0, 1]` corresponds to a minimum of the
expected value :math:`\mathbf{E}_y {\mathcal{L}_\tau(y, y_\tau)}` of the quantile
loss function

.. math::

   \mathcal{L}_\tau(y, y_\tau) =
      \begin{cases}
      \tau (y - y_\tau) & \text{if } y > y_\tau \\
      (1 - \tau) (y_\tau - y) & \text{otherwise}.
      \end{cases}

A proof of this can be found on `wikipedia <https://en.wikipedia.org/wiki/Quantile_regression>`_.

Because of this property, training a neural network using the quantile loss
function :math:`\mathcal{L}_\tau` will teach the network to predict the
corresponding quantile :math:`y_\tau`. QRNNs extend this principle to a
sequence of quantiles corresponding to an arbitrary selection of quantile
fractions :math:`\tau_1, \tau_2, \ldots` which are optimized simultaneously.

Defining a model
----------------

To define a QRNN model you need to specify the quantiles to predict as well as the
architecture of the underlying network to use. The code snippet below shows how
to create a QRNN model to predict the first until 99th percentiles.

As neural network model the QRNN uses fully-connected neural network with four hidden
layers with 128 neurons each and ReLU activation functions. Since this is taken
from the simple example notebook, the number of input features is set to 1.

.. code ::

  from quantnn import QRNN
  quantiles = np.linspace(0.01, 0.99, 99)

  layers = 4
  neurons = 128
  activation = "relu"
  model = (layers, neurons, activation)

  qrnn = q.QRNN(quantiles, input_dimensions=1, model=model)

QRNN provides a simplified interface to create QRNNs with simple fully-connected
architectures. The QRNN class, however, doesn't restrict you to use a fully-connected
network but supports any other suitable architecture such as, for example, 
fully-convolutional DenseNet-type architecture:

.. code ::

  model = ... # Define model as Pytorch Module or Keras model object.
  qrnn = q.QRNN(quantiles, model)

Training
--------

Evaluation
----------

Saving and loading
------------------

