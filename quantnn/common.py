"""
quantnn.common
==============

Implements common features used by the other submodules of the ``quantnn``
package.
"""
class QRNNException(Exception):
    """ Base exception for exception from the QRNN package."""

class UnknownArrayTypeException(QRNNException):
    """Thrown when a function is called with an unsupported array type."""

class UnknownModuleException(QRNNException):
    """
    Thrown when an unsupported backend is passed to a generic array
    operation.
    """

class InvalidDimensionException(QRNNException):
    """Thrown when an input array doesn't match expected shape."""
