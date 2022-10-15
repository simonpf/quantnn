"""
==================
quantnn.normalizer
==================

This module provides a simple normalizer class, which can be used
to normalize input data and store the normalization data.
"""
from abc import ABC, abstractmethod
import pickle

import numpy as np
from quantnn.files import read_file


class Identity:
    """
    A dummy normalizer that does nothing. Useful as default value.
    """

    def __call__(self, x):
        return x

    def invert(self, y):
        return y


class NormalizerBase(ABC):
    """
    The Normalizer class can be used to normalize input data to a neural
    network. On creation, it computes mean and standard deviation of the
    provided input data and stores it so that it can be applied to other
    datasets.
    """

    def __init__(self, x, exclude_indices=None, feature_axis=1):
        """
        Create Normalizer object for given input data.

        Arguments:
            x: Tensor containing the input data.
            exclude_indices: List of integer containing feature indices
                that should not be normalized.
            feature_axis: The axis along which the input features are located.
        """
        x = x.astype(np.float32)

        n = x.shape[feature_axis]
        self.stats = {}

        if exclude_indices is None:
            self.exclude_indices = []
        else:
            self.exclude_indices = exclude_indices
        self.feature_axis = feature_axis
        selection = [slice(0, None)] * len(x.shape)

        indices = [i for i in range(n) if i not in self.exclude_indices]
        for i in indices:
            selection[feature_axis] = i
            self.stats[i] = self._get_stats(x[tuple(selection)], i)

    @abstractmethod
    def _get_stats(self, x, index):
        """
        Extract normalization statistics for feature.

        Args:
            x: Slice containing the non-normalized feature data.
            index: The index of the feature along the feature axis.

        Return:
           An arbitrary object containing the relevant statistics required
           by the normalizer.
        """

    @abstractmethod
    def _normalize(self, x, stats, rng=None):
        """
        Normalize a feature slice using the corresponding statistics.

        This abstract method encapsulates the specific normalization method that
        should be applied to each input feature.

        Args:
            x: Slice containing the features data to normalize.
            stats: The stats information corresponding to the given feature
                that was previously extracted using the _get_stats method.

        Return:
           Array of the same size as x containing the normalized values.
        """

    @abstractmethod
    def _invert(self, x_normed, stats):
        """
        Undo normalization of a feature slice using the corresponding
        statistics.

        Args:
            x: Slice containing the normalized feature data to un-normalize.
            stats: The stats information corresponding to the given feature
                that was previously extracted using the _get_stats method.

        Return:
           Array of the same size as ``x_normed`` containing
           the un-normalized values.
        """

    def __call__(self, x, rng=None):
        """
        Applies normalizer to input data.

        Args:
            x: The input tensor to normalize.
            rng: Optional numpy random number generator, which will be
                used to randomize the replacement values for NAN inputs.

        Returns:
            The input tensor x normalized using the normalization data
            of this normalizer object.
        """
        normalized = []
        n = x.shape[self.feature_axis]
        selection = [slice(0, None)] * len(x.shape)

        dtype = x.dtype

        for i in range(n):
            selection[self.feature_axis] = i
            if i in self.stats:
                x_slice = x[tuple(selection)].astype(np.float32)
                x_normed = self._normalize(x_slice, self.stats[i], rng=rng)
            else:
                x_normed = x[tuple(selection)]
            x_normed = np.expand_dims(x_normed, self.feature_axis)
            normalized.append(x_normed.astype(np.float32))

        return np.concatenate(normalized, self.feature_axis).astype(dtype)

    def invert(self, x):
        """
        Reverses application of normalizer to given data.

        Args:
            x: The input tensor to denormalize.

        Returns:
            The input tensor x denormalized using the normalization data
            of this normalizer object.
        """
        inverted = []
        n = x.shape[self.feature_axis]
        selection = [slice(0, None)] * len(x.shape)

        dtype = x.dtype

        for i in range(n):
            selection[self.feature_axis] = i
            if i in self.stats:
                x_slice = x[tuple(selection)].astype(np.float32)
                x_inverted = self._invert(x_slice, self.stats[i])
            else:
                x_inverted = x[tuple(selection)]
            x_inverted = np.expand_dims(x_inverted, self.feature_axis)
            inverted.append(x_inverted.astype(np.float32))

        return np.concatenate(inverted, self.feature_axis).astype(dtype)

    @staticmethod
    def load(filename):
        """
        Load normalizer from file.

        Args:
            filename: The path to the file containing the normalizer.

        Returns:
            The loaded Normalizer object
        """
        with read_file(filename, "rb") as file:
            return pickle.load(file)

    def save(self, filename):
        """
        Store normalizer to file.

        Saves normalizer to a file using pickle.

        Args:
            filename: The file to which to store the normalizer.
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)


class Normalizer(NormalizerBase):
    """
    The Normalizer class can be used to normalize input data to a neural
    network. On creation, it computes mean and standard deviation of the
    provided input data and stores it so that it can be applied to other
    datasets.
    """

    def __init__(self, x, exclude_indices=None, feature_axis=1):
        """
        Create Normalizer object for given input data.

        Arguments:
            x: Tensor containing the input data.
            exclude_indices: List of integer containing feature indices
                that should not be normalized.
            feature_axis: The axis along which the input features are located.
        """
        super().__init__(x, exclude_indices=exclude_indices, feature_axis=feature_axis)

    def _get_stats(self, x, index):
        mean = x.mean()
        std_dev = x.std()
        return (mean, std_dev)

    def _normalize(self, x_slice, stats, rng=None):
        mean, std_dev = stats
        if np.isclose(std_dev, 0.0):
            x_normed = -1.0 * np.ones_like(x_slice)
        else:
            x_normed = (x_slice - mean) / std_dev
        return x_normed

    def _invert(self, x_slice, stats):
        mean, std_dev = stats
        if np.isclose(std_dev, 0.0):
            x_inverted = mean * np.ones_like(x_slice)
        else:
            x_inverted = (x_slice * std_dev) + mean
        return x_inverted


class MinMaxNormalizer(NormalizerBase):
    """
    The Normalizer class can be used to normalize input data to a neural
    network. On creation, it computes mean and standard deviation of the
    provided input data and stores it so that it can be applied to other
    datasets.
    """

    def __init__(self, x, exclude_indices=None, feature_axis=1, replace_nan=True):
        """
        Create Normalizer object for given input data.

        Arguments:
            x: Tensor containing the input data.
            exclude_indices: List of integer containing feature indices
                that should not be normalized.
            feature_axis: The axis along which the input features are located.
        """
        super().__init__(x, exclude_indices=exclude_indices, feature_axis=feature_axis)
        self.replace_nan = replace_nan

    def _get_stats(self, x, index):
        if np.any(np.isfinite(x)):
            x_min = np.nanmin(x)
            x_max = np.nanmax(x)
        else:
            x_min = np.nan
            x_max = np.nan
        return (x_min, x_max)

    def _normalize(self, x_slice, stats, rng=None):
        x_min, x_max = stats
        d_x = x_max - x_min

        l = -1.0
        r = 1.0

        if np.isclose(d_x, 0.0):
            x_normed = -1.0 * np.ones_like(x_slice)
        else:
            x_normed = l + (r - l) * (x_slice - x_min) / d_x
            x_normed = np.maximum(np.minimum(x_normed, 1.0), -1.0)

        if self.replace_nan:
            missing = -1.5
            if rng is not None:
                missing = missing * rng.uniform(0.95, 1.05)
            x_normed[np.isnan(x_slice)] = missing

        return x_normed

    def _invert(self, x_slice, stats):
        x_min, x_max = stats
        d_x = x_max - x_min

        l = -1.0
        r = 1.0
        indices = x_slice <= -1.5

        if np.isclose(d_x, 0.0):
            x_inverted = x_min * np.ones_like(x_slice)
        else:
            x_inverted = ((x_slice - l) / (r - l) * d_x) + x_min

        if self.replace_nan:
            x_inverted[indices] = np.nan

        return x_inverted
