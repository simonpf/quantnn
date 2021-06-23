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


class UnsupportedTensorType(QuantnnException):
    """
    Thrown when quantnn is asked to handle a tensor type it doesn't support.
    """


class UnknownModuleException(QuantnnException):
    """
    Thrown when an unsupported backend is passed to a generic array
    operation.
    """


class UnsupportedBackendException(QuantnnException):
    """
    Thrown when quantnn is requested to load a backend that is not supported.
    """


class MissingBackendException(QuantnnException):
    """
    Thrown when a requested backend could not be imported.
    """


class InvalidDimensionException(QuantnnException):
    """Thrown when an input array doesn't match expected shape."""


class ModelNotSupported(QuantnnException):
    """Thrown when a provided model isn't supported by the chosen backend."""


class MissingAuthenticationInfo(QuantnnException):
    """Thrown when required authentication information is not available."""


class DatasetError(QuantnnException):
    """
    Thrown when a given dataset object does not provide the expected interface.
    """


class InvalidURL(QuantnnException):
    """
    Thrown when a provided file URL is invalid.
    """


class InputDataError(QuantnnException):
    """
    Thrown when the training data does not match the expected format.
    """

class ModelLoadError(QuantnnException):
    """
    Thrown when an error occurs while a model is loaded.
    """
