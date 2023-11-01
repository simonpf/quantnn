"""
quantnn.models.pytorch.factories
================================

Factory objects to build neural networks.
"""
from torch import nn
from . import masked as nm

class MaxPooling:
    """
    Factory for creating max-pooling downsampling layers.
    """
    def __init__(
            self,
            kernel_size=(2, 2),
            project_first=True,
            masked=False
    ):
        """
        Args:
            project_first: If the channel during the downsampling
                are increased and project_first is True, channel
                numbers are increased prior to the downsampling
                operation.

        """
        self.kernel_size = kernel_size
        self.project_first = project_first
        self.masked = masked

    def __call__(
            self,
            channels_in,
            channels_out,
            f_down
    ):
        """
        Args:
            channels_in: The number of input channels.
            channels_out: The number of input channels.
            f_down: The number of output channels.
        """
        if self.masked:
            mod = nm
        else:
            mod = nn

        pool = mod.MaxPool2d(
            kernel_size=self.kernel_size,
            stride=f_down
        )

        if channels_in == channels_out:
            return pool

        project = mod.Conv2d(channels_in, channels_out, kernel_size=1)

        if self.project_first:
            return nn.Sequential(
                project,
                pool
            )
        return nn.Sequential(
            pool,
            project
        )
