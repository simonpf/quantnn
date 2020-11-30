"""
qrnn
====

The ``qrnn`` packages provides an implementation of quantile regression neural
networks on top of the PyTorch and Keras machine learning packages.
"""
from quantnn.qrnn import QRNN, set_backend, get_backend
from quantnn.functional import (cdf,
                                pdf,
                                posterior_mean,
                                probability_less_than,
                                probability_larger_than,
                                sample_posterior,
                                sample_posterior_gaussian)
