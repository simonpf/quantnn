.. quantnn documentation master file, created by
   sphinx-quickstart on Sun Dec  6 10:44:55 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

quantnn
=======

**quantnn** is a Python package for solving probabilistic regression problems using
(deterministic) deep neural networks, i.e. to predict a conditional distribution
:math:`P(y|x)` for given input :math:`x`.

It currently provides implementations of two neural-network based methods to solve
these type of problems:

1. **Quantile regression neural networks (QRNNs)**: QRNNs learn to predict the quantiles
   of the conditional distribution :math:`P(y|x)`, which can be used to estimate its
   cumulative distribution function (CDF).

   .. figure:: cdf_qrnn.png
    :width: 800
    :alt: A conditional Cumulative distribution function predicted using a QRNN.

    Example of a QRNN applied to predict the quantiles of a function with
    heteroscedastic noise.

2. **Density regression neural networks (DRNNs)**: DRNNs learn  predict a binned
   version of the probability density function (PDF) of :math:`P(y|x)`.

   .. figure:: pdf_drnn.png
      :width: 800
      :alt: A conditional probability density function predicted using a DRNN.

      Example of a DRNN applied to predict the quantiles of a function with
      heteroscedastic noise.



Features
--------

- A flexible, high-level implementation of QRNN and DRNNs supporting both PyTorch and Keras
  as backends.
- Generic functions to manipulate and process QRNN and DRNN predictions such as computing the
  posterior mean or classifying inputs.

Installation
------------

The currently recommended way of installing the **quantnn** package is to checkout the source from
`GitHub <http://github.com/simonpf/quantnn>`_ and install in editable mode using ``pip``:

.. code-block:: bash

   pip install -e .

Content
-------

.. toctree::
   :maxdepth: 2

   user_guide
   examples
   api_reference



