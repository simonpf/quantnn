"""
quantnn.models.pytorch.blocks
=============================

Provides factories for various block types.
"""
from typing import Optional, Callable

import torch
from torch import nn


from quantnn.models.pytorch.base import ParamCount
from . import normalization
from . import downsampling
from . import masked as nm


class ConvBlockFactory:
    """
    A basic convolution block.

    This basic convolution block combines a convolution with an optional
    normalization layer and activation function.
    """

    def __init__(
        self, kernel_size=1, norm_factory=None, activation_factory=None, masked=False
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
        self.masked = masked

    def __call__(
        self,
        channels_in: int,
        channels_out: Optional[int] = None,
        downsample: Optional[int] = None,
        block_index: int = 0,
        **kwargs
    ):
        """
        Args:
            channels_in: The number of channels in the block input.
            channels_out: The number of channels in the block output. If not
                given, will be the same as number of input channels.
            downsample: Optional downsampling factor that will be used as the
                the stride of the convolution operation.
            block_index: Not used by this factory.
        """
        if self.masked:
            mod = nm
        else:
            mod = nn

        if channels_out is None:
            channels_out = channels_in

        padding = 0
        if self.kernel_size > 1:
            padding = (self.kernel_size - 1) // 2

        if downsample is None:
            downsample = 1

        blocks = [
            mod.Conv2d(
                channels_in,
                channels_out,
                kernel_size=self.kernel_size,
                padding=padding,
                stride=downsample,
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

    def __init__(self, kernel_size=1, norm_factory=None, activation_factory=None):
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
        channels_in: int,
        channels_out: Optional[int] = None,
        downsample: Optional[int] = None,
        block_index: int = 0,
    ):
        """
        Args:
            channels_in: The number of channels in the block input.
            channels_out: The number of channels in the block output. If not
                given, will be the same as number of input channels.
            downsample: Optional downsampling factor that will be used as the
                the stride of the convolution operation.
            block_index: Not used by this factory.
        """
        if channels_out is None:
            channels_out = channels_in

        padding = 0
        if self.kernel_size > 1:
            padding = (self.kernel_size - 1) // 2

        if downsample is None:
            downsample = 1

        blocks = [
            nn.ConvTranspose2d(
                channels_in,
                channels_out,
                kernel_size=self.kernel_size,
                padding=padding,
                stride=downsample,
            )
        ]

        if self.norm_factory is not None:
            blocks.append(self.norm_factory(channels_out))
        if self.activation_factory is not None:
            blocks.append(self.activation_factory())

        if len(blocks) == 1:
            return blocks[0]

        return nn.Sequential(*blocks)


class ResNeXtBlock(nn.Module, ParamCount):
    """
    A convolutional block modeled after the ResNeXt architecture.
    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        bottleneck: int = 2,
        cardinality: int = 32,
        projection: Optional[nn.Module] = None,
        stride: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = nn.ReLU,
        masked: bool = False,
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

        mod = nn
        if masked:
            mod = nm

        if norm_layer is None:
            norm_layer = mod.BatchNorm2d

        self.conv_1 = mod.Conv2d(channels_in, channels_out // bottleneck, kernel_size=1)
        self.norm_1 = norm_layer(channels_out // bottleneck)
        self.conv_2 = mod.Conv2d(
            channels_out // bottleneck,
            channels_out // bottleneck,
            kernel_size=3,
            groups=cardinality,
            stride=stride,
            padding=1,
        )
        self.norm_2 = norm_layer(channels_out // bottleneck)
        self.conv_3 = mod.Conv2d(
            channels_out // bottleneck,
            channels_out,
            kernel_size=1,
        )
        self.norm_3 = norm_layer(channels_out)
        self.act = activation()
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
        norm_factory: Optional[Callable[[int], nn.Module]] = None,
        activation_factory: Optional[Callable[[int], nn.Module]] = None,
        masked: bool = False,
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
        self.norm_factory = norm_factory
        self.masked = masked

        if masked:
            mod = nm
        else:
            mod = nn

        if activation_factory is None:
            activation_factory = nn.ReLU
        self.activation_factory = activation_factory

        if norm_factory is None:
            norm_factory = mod.BatchNorm2d
        self.norm_factory = norm_factory

    def __call__(
        self,
        channels_in: int,
        channels_out: int,
        downsample: Optional[int] = None,
        block_index: int = 0,
        **kwargs
    ) -> nn.Module:
        """
        Create ResNeXt block.

        Args:
            channels_in: The number of channels in the input.
            channels_out: The number of channels used in the remaining
                layers in the block.
            downsample: Degree of downsampling to be performed by the block.
                No downsampling is performed if <= 1.
            block_index: Not used by this factory.

        Return:
            The ResNeXt block.
        """
        mod = nn
        if self.masked:
            mod = nm

        stride = (1, 1)
        projection = None

        if isinstance(downsample, int):
            downsample = (downsample,) * 2

        if downsample is not None and max(downsample) > 1:
            stride = downsample
        if max(stride) > 1 or channels_in != channels_out:
            if max(stride) > 1:
                projection = nn.Sequential(
                    mod.AvgPool2d(kernel_size=stride, stride=stride),
                    mod.Conv2d(channels_in, channels_out, kernel_size=1),
                )
            else:
                projection = nn.Sequential(
                    mod.Conv2d(channels_in, channels_out, kernel_size=1),
                )

        return ResNeXtBlock(
            channels_in,
            channels_out,
            stride=stride,
            projection=projection,
            cardinality=self.cardinality,
            norm_layer=self.norm_factory,
            activation=self.activation_factory,
            masked=self.masked,
        )


class ConvNextBlock(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: Optional[int] = None,
        kernel_size: int = 7,
        layer_scale: float = 1e-6,
        activation_factory: Callable[[], nn.Module] = nn.GELU,
        norm_factory: Callable[[int], nn.Module] = normalization.LayerNormFirst,
        channel_scaling: int = 4,
    ):
        """
        A ConvNext block as defined in 'A ConvNet for the 2020s'.

        The numbers of incoming and outgoing channels are allowed to differ.
        If that is the case, an additional point-wise convolution layer is
        used to adapt the number of channels priort to feeding the input into
        the block.

        Args:
            channels_in: The number of incoming channels.
            channels_out: The number of channes returned from the block.
            kernel_size: The kernel size used for the first convolution of
                the block.
            layer_scale: Initial value for the scaling of the block outputs
                that is applied before the skip connection.
            activation_factory: Factory functional to create the activation
                function used in the block. Defaults to GELU, which is proposed
                in the original implementation.
            norm_factory: Normalization functional to create the normalization
                module used in the block. Defaults to layer norm, which is
                proposed in the original implementation.
            channel_scaling: The channel scaling applied in the inverted
                bottleneck. Defaults to 4, which is the value used in the
                original implementation.
        """
        super().__init__()

        if channels_out is None:
            channels_out = channels_in

        padding = kernel_size // 2
        modules = []
        if channels_out != channels_in:
            modules.append(nn.Conv2d(channels_in, channels_out, kernel_size=1))
            self.proj = nn.Conv2d(channels_in, channels_out, kernel_size=1)
        else:
            self.proj = nn.Identity()
        modules += [
            nn.Conv2d(
                channels_out,
                channels_out,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels_out,
            ),
            norm_factory(channels_out),
            nn.Conv2d(
                channels_out,
                channel_scaling * channels_out,
                kernel_size=1,
            ),
            activation_factory(),
            nn.Conv2d(channel_scaling * channels_out, channels_out, kernel_size=1),
        ]
        self.body = nn.Sequential(*modules)

        self.layer_scale = None
        if layer_scale > 0:
            self.layer_scale = nn.Parameter(
                layer_scale * torch.ones((channels_out, 1, 1)), requires_grad=True
            )

    def forward(self, x):
        """
        Propagate x through block.
        """
        y = self.body(x)
        if self.layer_scale is not None:
            y = self.layer_scale * y
        return self.proj(x) + y


class ConvNextBlockV2(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: Optional[int] = None,
        kernel_size: int = 7,
        activation_factory: Callable[[], nn.Module] = nn.GELU,
        norm_factory: Callable[[int], nn.Module] = normalization.LayerNormFirst,
        channel_scaling: int = 4,
    ):
        """
        Updated ConvNext block as defined in
        'ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders'.

        Args:
            channels_in: The number of incoming channels.
            channels_out: The number of channes returned from the block.
            kernel_size: The kernel size used for the first convolution of
                the block.
            activation_factory: Factory functional to create the activation
                function used in the block. Defaults to GELU, which is proposed
                in the original implementation.
            norm_facttory: Normalization functional to create the normalization
                module used in the block. Defaults to layer norm, which is
                proposed in the original implementation.
            channel_scaling: The channel scaling applied in the inverted
                bottleneck. Defaults to 4, which is the value used in the
                original implementation.
        """
        super().__init__()

        if channels_out is None:
            channels_out = channels_in

        modules = []

        if channels_out != channels_in:
            modules.append(nn.Conv2d(channels_in, channels_out, kernel_size=1))
            self.proj = nn.Conv2d(channels_in, channels_out, kernel_size=1)
        else:
            self.proj = nn.Identity()

        padding = (kernel_size - 1) // 2

        modules += [
            nn.Conv2d(
                channels_out,
                channels_out,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels_out,
            ),
            norm_factory(channels_out),
            nn.Conv2d(
                channels_out,
                channel_scaling * channels_out,
                kernel_size=1,
            ),
            normalization.GRN(channel_scaling * channels_out),
            activation_factory(),
            nn.Conv2d(channel_scaling * channels_out, channels_out, kernel_size=1),
        ]
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        """
        Propagate x through block.
        """
        y = self.body.forward(x)
        return self.proj(x) + y


class ConvNextBlockFactory:
    """
    Factory class to generate ConvNext blocks.
    """

    def __init__(
        self,
        version: int = 1,
        kernel_size: int = 7,
        activation_factory: Callable[[], nn.Module] = nn.GELU,
        norm_factory: Callable[[int], nn.Module] = normalization.LayerNormFirst,
        channel_scaling: int = 4,
        masked: bool = False,
    ):
        """
        version: Which version of ConvNext blocks should be created. Should be
            '1' or '2'.
        kernel_size: The kernel size used for the first convolution of
            the block.
        activation_factory: Factory functional to create the activation
            function used in the block. Defaults to GELU, which is proposed
            in the original implementation.
        norm_factory: Normalization functional to create the normalization
            module used in the block. Defaults to layer norm, which is
            proposed in the original implementation.
        channel_scaling: The channel scaling applied in the inverted
            bottleneck. Defaults to 4, which is the value used in the
            original implementation.
        """
        self.version = version
        self.kernel_size = kernel_size
        self.activation_factory = activation_factory
        self.norm_factory = norm_factory
        self.channel_scaling = channel_scaling
        self.masked = masked

    def __call__(
        self,
        channels_in: int,
        channels_out: Optional[int] = None,
        downsample: Optional[int] = None,
        block_index: int = 0,
        **kwargs
    ):
        """
        Create ConvNext block.

        Args:
            channels_in: The number of channels in the input.
            channels_out: The number of channels used in the remaining
                layers in the block.
            downsample: Degree of downsampling to be performed by the block.
                No downsampling is performed if <= 1.
            block_index: Not used by this factory.

        Return:
            The ConvNext block.
        """

        block_in = channels_in
        if downsample is not None:
            block_in = channels_out

        if self.version == 1:
            block = ConvNextBlock(
                channels_in=block_in,
                channels_out=channels_out,
                kernel_size=self.kernel_size,
                activation_factory=self.activation_factory,
                norm_factory=self.norm_factory,
                channel_scaling=self.channel_scaling,
            )
        else:
            block = ConvNextBlockV2(
                channels_in=block_in,
                channels_out=channels_out,
                kernel_size=self.kernel_size,
                activation_factory=self.activation_factory,
                norm_factory=self.norm_factory,
                channel_scaling=self.channel_scaling,
            )

        if downsample is None or not downsample:
            return block

        fac = downsampling.ConvNextDownsamplerFactory()
        return nn.Sequential(fac(channels_in, channels_out, downsample), block)
