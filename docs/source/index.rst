.. quantnn documentation master file, created by
   sphinx-quickstart on Sun Dec  6 10:44:55 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

quantnn
=======

The **quantnn** package provides an implementation of **quantile regression neural
networks (QRNNs)** in Python. QRNNs can be used to estimate the epistemic uncertainty
in a regression task by learning to predict the quantiles :math:`y_\tau` of the conditional
distribution :math:`p(y | \mathbf{x})`.

.. figure:: quantiles.svg
   :width: 600
   :alt: Plot of empirical and predicted quantiles.

   Example of a QRNN applied to predict the quantiles of a function :math:`y` with heteroscedastic
   noise.



Features
--------

- A flexible, high-level implementation of QRNNs currently supporting PyTorch and Keras (Tensorflow)
  as backends.
- Generic functions to manipulate and process quantile predictions such as computing the posterior mean
  or classifying inputs.

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



