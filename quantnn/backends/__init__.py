from quantnn.backends.pytorch import PyTorch
from quantnn.common import UnsupportedTensorType

_TENSOR_BACKEND_CLASSES = [PyTorch]

TENSOR_BACKENDS = [b for b in _TENSOR_BACKEND_CLASSES if b.available()]


def get_tensor_backend(tensor):
    """
    Determine the tensor backend for a given tensor.

    Args:
        tensor: A tensor type of any of the supported backends.

    Return:
        The backend class which providing the interface to the tensor library
        corresponding to ``tensor``.

    Raises:
        :py:class:`~quantnn.common.UnsupportedTensorType` when the tensor type is not
        supported by quantnn.
    """
    for backend in TENSOR_BACKENDS:
        if backend.matches_tensor(tensor):
            return backend
    raise UnsupportedTensorType(
        f"The provided tensor of type {type(tensor)} is not supported by quantnn."
    )
