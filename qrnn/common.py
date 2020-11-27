"""
qrnn.common
===========

Implements common features used by the other submodules of the ``qrnn``
package.
"""

class QRNNException(Exception):
    """ Base exception for exception from the QRNN package."""

class UnknownArrayTypeException(QRNNException):
    """Thrown when a function is called with an unsupported array type."""

class InvalidDimensionException(QRNNException):
    """Thrown when an input array doesn't match expected shape."""
