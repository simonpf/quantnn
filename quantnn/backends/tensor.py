"""
=======================
quantnn.backends.tensor
=======================

This module defines an abstract base class defining the general interface
for tensor backend module.
"""
from abc import ABC, abstractclassmethod


class TensorBackend(ABC):
    @abstractclassmethod
    def available(cls):
        """
        Determine whether this backend is available.

        Returns:
            ``True`` if the backend is available, ``False`` otherwise.
        """

    @abstractclassmethod
    def matches_tensor(cls, tensor):
        """
        Determine whether a given tensor belongs to a given TensorBackend
        class.

        Args:
            tensor: The tensor for which to determine whether it belongs to the
                 given backend.

        Returns:
            ``True`` when the given tensor belongs to the tensor backend.
        """

    @abstractclassmethod
    def from_numpy(cls, array, like=None):
        """
        Creake a rank-k tensor from a rank-k numpy array.

        Args:
            array: The input data to turn into a tensor of the respective
                backend.
            like: If provided used to determine additional characteristics
                of the tensor, such as data type and device.

        Returns:
            A rank-k tensor containing the data in ``array`` with the same
            characteristics as the tensor ``like``.
        """

    @abstractclassmethod
    def to_numpy(cls, array):
        """
        Convert a rank-k tensor to numpy array.

        Args:
            array: The backend tensor to turn into a numpy array.

        Returns:
            Numpy array containing the data of the given backend tensor.
        """

    @abstractclassmethod
    def as_type(cls, tensor, like):
        """
        Convert tensor to match another tensor in data type and other
        characteristics.

        This function will convert ``tensor`` to a format compatible to
        ``like`` so that they can be used together in numerical operations.

        Args:
             tensor: The tensor whose datatype to convert.
             like: A tensor object defining the target data type.

        Returns:

            A new tensor containing the data of ``tensor`` casted to the
            datatype of ``like``.
        """

    @abstractclassmethod
    def to_tensor(cls, tensor, like=None):
        """
        Convert a tensor to a tensor of the given tensor backend.

        If the tensor is from another tensor backend, it will be converted
        to the tensor backend corresponding to this class by converting
        it to a numpy array and from that to a tensor of this backend.

        If the tensor is already a tensor of this backend, it will directly
        return the tensor, or, if ``like`` is given, convert it to a tensor
        that is compatible with the tensor ``like``.
        """
        from quantnn.backends import get_tensor_backend

        if type(tensor).__module__ in ["numpy", "numpy.ma"]:
            return cls.from_numpy(tensor, like)

        backend = get_tensor_backend(tensor)
        if backend == cls:
            if like is not None:
                return cls.as_type(tensor, like)
            else:
                return tensor
        array = backend.to_numpy(tensor)
        return cls.from_numpy(array, like)

    @abstractclassmethod
    def sample_uniform(cs, shape=None, like=None):
        """
        Generate a tensor containing samples from a uniform distribution over [0, 1].

        Args:
             shape: Tuple describing the shape of the tensor.
             like: If provided, used to determine dtype, device and other
                  relevant properties of the result tensor. Only used to
                  determine the output shape if ``shape`` is ``None``.

        Returns:
             Tensor of the required shape containing sample from a uniform
             distribution.
        """

    @abstractclassmethod
    def sample_gaussian(cs, shape=None, like=None):
        """
        Generate a tensor containing samples from a Normal distribution with 0 mean
        and unit standard deviation.

        Args:
             shape: Tuple describing the shape of the tensor.
             like: If provided, used to determine dtype, device and other
                  relevant properties of the result tensor. Only used to
                  determine the output shape if ``shape`` is ``None``.

        Returns:
             Tensor of the required shape containing sample from a uniform
             distribution.
        """

    @abstractclassmethod
    def size(cs, tensor):
        """
        Args:
            tensor: The tensor of which to determine the number of elements.

        Return:
            The number of elements in a tensor.
        """

    @abstractclassmethod
    def concatenate(cls, tensors, dimension):
        """
        Concatenate a list of tensors along a given dimension.

        Args:
            tensors: A list containing the tensors to concatenate.
            dimension: The index of the dimension along which to concatenate
                the tensors.

        Returns:
            The tensor resulting from concatenating the tensors in ``tensors``
            along the given dimension.
        """

    @abstractclassmethod
    def expand_dims(cls, tensor, dimension_index):
        """
        Expand the rank of a tensor along a given dimension.

        This function inserts a dummy dimension of length 1 along the given
        position of the dimension-tuple describing the shape of the tensor.

        Args:
            tensor: The rank-k tensor whose dimensions to expand.
            dimension: The index in the shape-tuple of the tensor at which
                 to insert the dummy dimension.

        Returns:
            A tensor of rank k + 1 containing the same data as the input
            tensor but with an additional dummy dimension inserted at the
            given position.
        """

    @abstractclassmethod
    def exp(cls, tensor):
        """
        Element-wise exponential.
        Args:
            tensor: The rank-k tensor of which to compute the exponential.

        Returns:
            Tensor of same rank as ``tensor`` containing the exponential of
            all elements in ``tensor``.
        """

    @abstractclassmethod
    def log(cls, tensor):
        """
        Element-wise natural logarithm.
        Args:
            tensor: The rank-k tensor of which to compute the logarithm.

        Returns:
            Tensor of same rank as ``tensor`` containing the natural logarithm
            of all elements in ``tensor``.
        """

    @abstractclassmethod
    def pad_zeros(cls, tensor, n, dimension):
        """
        Pad a given tensor with zeros along a given dimension.

        Args:
             tensor: The rank-k tensor to pad with zeros.
             n: The number of zeros to pad the tensor with at each end along
                  the given dimension.
             dimension: The index of the dimension along which to pad the
                  zeros.

        Returns:
             A new rank-k tensor containing the same data as ``tensor`` but
             with the requested number of ``0``s padded to each side along
             the specified dimension.
        """

    @abstractclassmethod
    def pad_zeros_left(cls, tensor, n, dimension):
        """
        Pad a given tensor with zeros on the left side along a given dimension.

        Args:
             tensor: The rank-k tensor to pad with zeros.
             n: The number of zeros to pad the tensor with along the
                 given dimension.
             dimension: The index of the dimension along which to pad the
                  zeros.

        Returns:
             A new rank-k tensor containing the same data as ``tensor`` but
             with the requested number of ``0``s padded to the left side along
             the specified dimension.
        """

    @abstractclassmethod
    def arange(cls, start, end, step, like=None):
        """
        Crate a rank-1 tensor containing a stepped sequence of values.

        Args:
            start: Start value of the sequence.
            end: Maximum value of the sequence.
            step: Step size.
            like: If provided, will be used to determine data type and device
                of the created tensor.

        Return:
            Rank-1 tensor containing a sequence starting with the given start
            value and increasing with the given step size up until the largest
            value that is strictly smaller than the given end value.
        """

    @abstractclassmethod
    def reshape(cls, tensor, shape):
        """
            Reshape a tensor.

            Arguments:
                tensor: The tensor to reshape.
                shape: Tuple containing the target shape to which to reshape
                    the tensor.

        Returns:
            The input ``tensor`` reshape into the requested shape.
        """

    @abstractclassmethod
    def trapz(cls, x, y, dimension):
        """
        Numerical integration using the trapezoidal rule.

        Note:
            This method requires ``y`` to contain the same number of elements
            as ``x`` along dimension ``dimension``.

        Args:
            x: Rank-1 tensor containing the abscissa values over which to
                integrate.
            y: Rank-k tensor containing the data to integrate.
            dimension: The index of the dimension over which to integrate.

        Returns:
            Rank k - 1 tensor containing the data in ``y`` integrated over
            ``x``.
        """

    @abstractclassmethod
    def integrate(cls, x, y, dimension):
        """
        Numerical integral of the tensor ``y`` over ``x`` along a given
        dimension.

        In contrast to the ``trapz`` method, this method works also for the
        case when ``x`` has one more element than ``y`` along the dimension
        specified by ``dimension``. In which ``y`` is interpreted as a
        piece-wise constant function over ``x``.

        If that is not the case, this method falls back to integration using
        the trapezoidal rule.

        Args:
            x: Rank-1 tensor containing the abscissa values over which to
                integrate.
            y: Rank-k tensor containing the data to integrate.
            dimension: The index of the dimension over which to integrate.

        Returns:
            Rank k - 1 tensor containing the data in ``y`` integrated over
            ``x``.
        """
        if len(x) == y.shape[dimension] + 1:
            dx = x[1:] - x[:-1]
            n = len(y.shape)
            dx_shape = [1] * n
            dx_shape[dimension] = -1
            dx = reshape(module, dx, dx_shape)
            return (y * dx).sum(dimension)
        return cls.trapz(x, y, dimension)

    @abstractclassmethod
    def cumsum(cls, y, dimension):
        """
        Calculate the cumulative sum along given axis.

        Arguments:
            y: Rank-k tensor to accumulate along given dimension.
            dimension: The dimension to sum over.

        Return:
            A rank-k tensor containing the cumulative sum calculated along
            the given dimension.
        """

    @abstractclassmethod
    def zeros(module, shape=None, like=None):
        """
        Zero tensor of given shape.

        Arguments:
            module: The backend array corresponding to the given array.
            shape: Tuple defining the desired shape of the tensor to create.
            like: Optional tensor to use to determine additional properties
                such as data type, device.

        Return:
            A tensor containing zeros of given shape.
        """

    @abstractclassmethod
    def ones(module, shape=None, like=None):
        """
        One tensor of given shape.

        Arguments:
            module: The backend array corresponding to the given array.
            shape: Tuple defining the desired shape of the tensor to create.
            like: Optional tensor to use to determine additional properties
                such as data type, device.

        Return:
            A tensor containing ones of the given shape.
        """

    @abstractclassmethod
    def softmax(module, x, axis=None):
        """
        Apply softmax to tensor.

        Arguments:
            module: The backend array corresponding to the given array.
            x: The tensor to apply the softmax to.

        Return:
            softmax(x)
        """

    @abstractclassmethod
    def where(module, condition, x, y):
        """
        Select from tensor 'x' or 'y' based on condition.

        Args:
            module: The backend array corresponding to the given array.
            condition: Rank-k bool tensor.
            x: Elements to pick from when condition is True.
            y: Elements to pick from when condition is False.

        Return:
            Rank-k tensor containing elements from 'x' where 'condition' is True
            and elements from 'y' everywhere else.
        """
