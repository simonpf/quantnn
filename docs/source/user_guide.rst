User guide
==========

This section describes the basic usage of the **quantnn** package.

Overview
--------

The main functionality of quantnn is implemented by two model classes:
:py:class:`quantnn.qrnn.QRNN` and :py:class:`quantnn.drnn.DRNN`. The
:py:class:`~quantnn.qrnn.QRNN` class provides an implementation of quantile
regression neural networks (QRNNs), whereas the :py:class:`~quantnn.drnn.DRNN`
implements density regression neural networks (DRNNs).

Basic workflow
--------------

The basic usage of both the  :py:mod:`~quantnn.QRNN` and :py:mod:`~quantnn.DRNN`
classes is similar and follows the generic machine learning workflow:

1. **Defining the model**: A model is defined by instantiating the
   corresponding model class. For this, you need to define the architecture of
   the underlying neural network and specify which quantiles to predict (QRNN)
   or the binning of the PDF (DRNN).

2. **Training the model**: The training phase is similar as for any other deep
   neural network. The neural network backend (PyTorch or Keras) takes care of
   the heavy lifting, you only have to choose the training parameters.

3. **Evaluating the model**: Finally, you will of course want to make
   predictions with you model. This is done using the model
   :py:meth:`~quantnn.QRNN.predict` method, which will produce a tensor of
   either quantiles (QRNN) or the binned PDF (DRNN) of the posterior
   distribution. To further process these prediction the
   :py:mod:`quantnn.quantiles` and :py:mod:`quantnn.density` module provide
   function that can be used to derive statistics of the probabilistic
   results.

4. **Loading and saving the model**: To reuse your train model you can save
   and load it using the corresponding class methods of the
   :py:class:`~quantnn.qrnn.QRNN` :py:class:`~quantnn.qrnn.DRNN` classes.

.. note ::

    Care has been taken to design the interfaces of the :py:class:`~quantnn.qrnn.QRNN`
    and :py:class:`~quantnn.drnn.DRNN` classes as consistently as possible so that
    both classes can be used interchangeably to the largest extent possible.

Content
-------

.. toctree::
   :maxdepth: 2
   
   qrnn
   drnn
   
   



