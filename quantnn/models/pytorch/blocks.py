"""
quantnn.models.pytorch.blocks
=============================

Provides factories for various block types.
"""
from typing import Optional, Callable

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
            downsample=1,
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

        if not downsample:
            downsample = 1

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

        if len(blocks) == 1:
            return blocks[0]

        return nn.Sequential(*blocks)


class ConvTransposedBlockFactory:
    """
    A convolution block using transposed convolutions.

    Combines transposed convolution with an optional normalization
    layer and activation function.
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
            downsample=1,
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

        if not downsample:
            downsample = 1

        blocks = [
            nn.ConvTranspose2d(
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

        if len(blocks) == 1:
            return blocks[0]

        return nn.Sequential(*blocks)


class ResNeXtBlock(nn.Module):
    """
    A convolutional block modeled after the ResNeXt architecture.
    """
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        bottleneck: int=2,
        cardinality: int=32,
        projection: Optional[nn.Module] = None,
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = nn.ReLU
    ):
        """
        Args:
            channels_in: The number of incoming channels.
            channels_out: The number of outgoing channels.
            bottleneck: Scaling factor defining the reduction in channels
                after the first layer of the block.
            projection: If given, a module that replaces the identity mapping
                prior to the addition with the residual pathway.
            stride: Stride applied in the 3x3 convolution layer.
            norm_layer: Factory to produce norm layers.
            activation: The activation function to use.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv_1 = nn.Conv2d(
            channels_in,
            channels_out // bottleneck,
            kernel_size=1
        )
        self.norm_1 = norm_layer(channels_out // bottleneck)
        self.conv_2 = nn.Conv2d(
            channels_out // bottleneck,
            channels_out // bottleneck,
            kernel_size=3,
            groups=cardinality,
            stride=stride,
            padding=1
        )
        self.norm_2 = norm_layer(channels_out // bottleneck)
        self.conv_3 = nn.Conv2d(
            channels_out // bottleneck,
            channels_out,
            kernel_size=1,
        )
        self.norm_3 = norm_layer(channels_out)
        self.act = activation(inplace=True)
        self.projection = projection


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward input tensor through block."""

        shortcut = x

        y = self.act(self.norm_1(self.conv_1(x)))
        y = self.act(self.norm_2(self.conv_2(y)))

        if self.projection is not None:
            shortcut = self.projection(x)

        y = self.norm_3(self.conv_3(y))
        y += shortcut

        return self.act(y)

class ResNeXtBlockFactory:
    """
    Factory wrapper for ``torchvision`` ResNeXt blocks.
    """

    def __init__(
            self,
            cardinality=32,
            norm_factory: Optional[Callable[[int], nn.Module]] = None
    ):
        """
        Args:
            cardinality: The cardinality of the block as defined
                in the ResNeXt paper.
            norm_factory: A factory object to produce the normalization
                layers used in the ResNet blocks. Defaults to batch
                norm.
        """
        self.cardinality = cardinality
        if norm_factory is None:
            norm_factory = nn.BatchNorm2d
        self.norm_factory = norm_factory

    def __call__(
        self, channels_in: int, channels_out: int, downsample: int = 1
    ) -> nn.Module:
        """
        Create ResNeXt block.

        Args:
            channels_in: The number of channels in the input.
            channels_out: The number of channels used in the remaining
                layers in the block.
            downsample: Degree of downsampling to be performed by the block.
                No downsampling is performed if <= 1.

        Return:
            The ResNeXt block.
        """
        stride = 1
        projection = None
        if downsample > 1:
            stride = downsample
        if downsample > 1 or channels_in != channels_out:
            if stride > 1:
                projection = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride),
                    nn.Conv2d(channels_in, channels_out, kernel_size=1),
                    self.norm_factory(channels_out),
                )
            else:
                projection = nn.Sequential(
                    nn.Conv2d(channels_in, channels_out, kernel_size=1),
                    self.norm_factory(channels_out),
                )

        return ResNeXtBlock(
            channels_in,
            channels_out,
            stride=stride,
            projection=projection,
            cardinality=self.cardinality,
            norm_layer=self.norm_factory
        )
