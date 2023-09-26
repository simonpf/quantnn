"""
quantnn.models.pytorch.downsampling
===================================

This module provides factory classes for downsampling modules.
"""
from typing import Union, Tuple

from torch import nn

from quantnn.models.pytorch.normalization import LayerNormFirst


class ConvNextDownsamplerFactory:
    """
    Downsampler factory consisting of layer normalization followed
    by strided convolution.
    """

    def __call__(
        self, channels_in: int, channels_out: int, f_dwn: Union[int, Tuple[int, int]]
    ):
        """
        Args:
            channels_in: The number of channels in the input.
            channels_out: The number of channels in the output.
            f_dwn: The downsampling factor. Can be a tuple if different
                downsampling factors should be applied along height and
                width of the image.
        """
        return nn.Sequential(
            LayerNormFirst(channels_in),
            nn.Conv2d(channels_in, channels_out, kernel_size=f_dwn, stride=f_dwn),
        )
