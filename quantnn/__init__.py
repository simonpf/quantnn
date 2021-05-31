r"""
=======
quantnn
=======

The quantnn package provides functionality for probabilistic modeling and prediction
using deep neural networks.

The two main features of the quantnn package are implemented by the
:py:class:`~quantnn.qrnn.QRNN` and :py:class:`~quantnn.qrnn.DRNN` classes, which implement
quantile regression neural networks (QRNNs) and density regression neural networks (DRNNs),
respectively.

The modules :py:mod:`quantnn.quantiles` and :py:mod:`quantnn.density` provide generic
(backend agnostic) functions to manipulate probabilistic predictions.
"""
import logging as _logging
import os

from rich.logging import RichHandler
from quantnn.neural_network_model import set_default_backend, get_default_backend
from quantnn.qrnn import QRNN
from quantnn.drnn import DRNN
from quantnn.quantiles import (
    cdf,
    pdf,
    posterior_mean,
    probability_less_than,
    probability_larger_than,
    sample_posterior,
    sample_posterior_gaussian,
    quantile_loss,
)

_LOG_LEVEL = os.environ.get("QUANTNN_LOG_LEVEL", "WARNING").upper()
_logging.basicConfig(
    level=_LOG_LEVEL, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
