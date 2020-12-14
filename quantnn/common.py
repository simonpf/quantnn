"""
quantnn.common
==============

Implements common features used by the other submodules of the ``quantnn``
package.
"""
class QuantnnException(Exception):
    """ Base exception for exception from the quantnn package."""


class UnknownArrayTypeException(QuantnnException):
    """Thrown when a function is called with an unsupported array type."""


class UnknownModuleException(QuantnnException):
    """
    Thrown when an unsupported backend is passed to a generic array
    operation.
    """

class UnsupportedBackendException(QuantnnException):
    """
    Thrown when quantnn is requested to load a backend that is not supported.

    """


class InvalidDimensionException(QuantnnException):
    """Thrown when an input array doesn't match expected shape."""
