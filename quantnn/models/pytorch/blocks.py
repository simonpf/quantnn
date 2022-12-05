"""
quantnn.models.pytorch.blocks
=============================

Provides factories for various block types.
"""
import torch
from torch import nn


class ConvBlockFactory:
    """
    A basic convolution block.

    This basic convolution block combines a convolution with an optional
    normalization layer and activation function.
    """
    def __init__(
            self,
            kernel_size=1,
            norm_factory=None,
            activation_factory=None
    ):
        """
        Args:
            kernel_size: The size of the convolution kernel.
            norm_factory: A factory to produce the normalization layer that is
                applied after the convolution.
            activation_factory: A factory to produce the activation function
                that applied after the convolution and the optional
                normalization layer.
        """
        self.kernel_size = kernel_size
        self.norm_factory = norm_factory
        self.activation_factory = activation_factory

    def __call__(
            self,
            channels_in,
            channels_out=None,
            downsample=1
    ):
        """
        Args:
            channels_in: The number of channels in the block input.
            channels_out: The number of channels in the block output. If not
                given, will be the same as number of input channels.
            downsample: Optional downsampling factor that will be used as the
                the stride of the convolution operation.
        """
        if channels_out is None:
            channels_out = channels_in

        padding = 0
        if self.kernel_size > 1:
            padding = (self.kernel_size - 1) // 2

        blocks = [
            nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size=self.kernel_size,
                padding=padding,
                stride=downsample
            )
        ]

        if self.norm_factory is not None:
            blocks.append(self.norm_factory(channels_out))
        if self.activation_factory is not None:
            blocks.append(self.activation_factory())

        return nn.Sequential(*blocks)
