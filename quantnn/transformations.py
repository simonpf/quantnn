"""
=======================
quantnn.transformations
=======================

This module defines transformations that can be applied to train
a network in a transformed space but evaluate it in the original
space.
"""
import numpy as np

from quantnn.backends import get_tensor_backend


class Log10:
    """
    Transforms values to log space.
    """

    def __init__(self):
        self.xp = None

    def __call__(self, x):
        """
        Transform tensor.

        Args:
            x: Tensor containing the values to transform.

        Return:
            Tensor containing the transformed values.

        """
        if self.xp is None:
            xp = get_tensor_backend(x)
            self.xp = xp
        else:
            xp = self.xp
        return xp.log(x.double()).float() / np.log(10)

    def invert(self, y):
        """
        Transform transformed values back to original space.

        Args:
            y: Tensor containing the transformed values to transform
                back.

        Returns:
            Tensor containing the original values.
        """
        if self.xp is None:
            xp = get_tensor_backend(y)
            self.xp = xp
        else:
            xp = self.xp
        return xp.exp(np.log(10) * y.double()).float()


class Log:
    """
    Transforms values to log space.
    """

    def __init__(self):
        self.xp = None

    def __call__(self, x):
        """
        Transform tensor.

        Args:
            x: Tensor containing the values to transform.

        Return:
            Tensor containing the transformed values.

        """
        if self.xp is None:
            xp = get_tensor_backend(x)
            self.xp = xp
        else:
            xp = self.xp
        return xp.log(x.double()).float()

    def invert(self, y):
        """
        Transform transformed values back to original space.

        Args:
            y: Tensor containing the transformed values to transform
                back.

        Returns:
            Tensor containing the original values.
        """

        if self.xp is None:
            xp = get_tensor_backend(y)
            self.xp = xp
        else:
            xp = self.xp
        return xp.exp(y.double()).float()


class Softplus:
    """
    Applies softplus transform to values.
    """

    def __init__(self):
        self.xp = None

    def __call__(self, x):
        """
        Transform tensor.

        Args:
            x: Tensor containing the values to transform.

        Return:
            Tensor containing the transformed values.

        """
        if self.xp is None:
            xp = get_tensor_backend(x)
            self.xp = xp
        else:
            xp = self.xp

        return xp.where(x > 10, x, xp.log(xp.exp(x) - 1.0 + 1e-30))

    def invert(self, y):
        """
        Transform transformed values back to original space.

        Args:
            y: Tensor containing the transformed values to transform
                back.

        Returns:
            Tensor containing the original values.
        """
        if self.xp is None:
            xp = get_tensor_backend(y)
            self.xp = xp
        else:
            xp = self.xp
        return xp.where(y > 10, y, xp.log(xp.exp(y) + 1.0))


class LogLinear:
    """
    Composition of natural logarithm transformation and for
    x > 1 and identity transformation for x <= 1.
    """

    def __init__(self):
        self.xp = None

    def __call__(self, x):
        """
        Transform tensor.

        Args:
            x: Tensor containing the values to transform.

        Return:
            Tensor containing the transformed values.

        """
        if self.xp is None:
            xp = get_tensor_backend(x)
            self.xp = xp
        else:
            xp = self.xp

        return xp.where(x > 1, x - 1.0, xp.log(x))

    def invert(self, y):
        """
        Transform transformed values back to original space.

        Args:
            y: Tensor containing the transformed values to transform
                back.

        Returns:
            Tensor containing the original values.
        """

        if self.xp is None:
            xp = get_tensor_backend(y)
            self.xp = xp
        else:
            xp = self.xp
        return xp.where(y > 0, y + 1.0, xp.exp(y))


class Id:
    """
    Identity transform for testing.
    """

    def __init__(self):
        self.xp = None

    def __call__(self, x):
        """
        Transform tensor.

        Args:
            x: Tensor containing the values to transform.

        Return:
            Tensor containing the transformed values.

        """
        return x

    def invert(self, y):
        """
        Transform transformed values back to original space.

        Args:
            y: Tensor containing the transformed values to transform
                back.

        Returns:
            Tensor containing the original values.

        """
        return y
