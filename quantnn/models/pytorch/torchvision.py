"""
quantnn.models.pytorch.torchvision
==================================

Wrapper for models defined in torchvision models.

Using this module obviously requires torchvision to be installed.
"""
from copy import copy
from functools import partial
from typing import Optional, Callable

from torch import nn
from torchvision import models
from torchvision.ops import Permute


class ResNetBlockFactory:
    """
    Factory wrapper for ``torchvision`` ResNet blocks.
    """

    def __init__(self, norm_factory: Optional[Callable[[int], nn.Module]] = None):
        """
        Args:
            norm_factory: A factory object to produce the normalization
                layers used in the ResNet blocks. Defaults to batch
                norm.
        """
        if norm_factory is None:
            norm_factory = nn.BatchNorm2d
        self.norm_factory = norm_factory

    def __call__(
        self, channels_in: int, channels_out: int, downsample: int = 1
    ) -> nn.Module:
        """
        Create ResNet block.

        Args:
            channels_in: The number of channels in the input.
            channels_out: The number of channels used in the remaining
                layers in the block.
            downsample: Degree of downsampling to be performed by the block.
                No downsampling is performed if <= 1.

        Return:
            The ResNet block.
        """
        stride = 1
        projection = None
        if downsample > 1:
            stride = downsample
        if downsample > 1 or channels_in != channels_out:
            projection = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, stride=stride, kernel_size=1),
                self.norm_factory(channels_out),
            )

        return models.resnet.BasicBlock(
            channels_in,
            channels_out,
            stride=stride,
            downsample=projection,
            norm_layer=self.norm_factory
        )


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
        self.resnext_class = copy(models.resnet.Bottleneck)
        # Increase width of bottleneck.
        self.resnext_class.expansion = 2

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

        planes = channels_out // self.resnext_class.expansion // self.cardinality
        return self.resnext_class(
            channels_in,
            planes,
            stride=stride,
            downsample=projection,
            groups=self.cardinality,
            norm_layer=self.norm_factory
        )


class ConvNeXtBlockFactory:
    """
    Factory wrapper for ``torchvision`` ConvNeXt blocks.
    """
    @classmethod
    def layer_norm_with_permute(cls, channels_in):
        """
        Layer norm with permutation of channel dimension for application
        in a CNN.
        """
        return nn.Sequential(
            Permute((0, 2, 3, 1)),
            cls.layer_norm(channels_in),
            Permute((0, 3, 1, 2))
        )

    @classmethod
    def layer_norm(cls, channels_in):
        """
        Layer norm with eps=1e-6.
        """
        return nn.LayerNorm(channels_in, eps=1e-6)

    def __init__(
        self,
        norm_factory: Optional[Callable[[int], nn.Module]] = None,
        layer_scale: float = 1e-6,
        stochastic_depth_prob: float = 0.0,
    ):
        """
        Args:
            norm_factory: A factory object to produce the normalization
                layers used in the ConvNeXt blocks. NOTE: The norm layer
                should normalize along the last dimension because the
                ConvNeXt blocks permute the input dimensions.
            layer_scale: Layer scaling applied to the residual block.
            stochastic_depth_prob: Probability of the residual block being
                removed during training.
        """
        if norm_factory is None:
            norm_factory = self.layer_norm
        self.norm_factory = norm_factory
        self.layer_scale = layer_scale
        self.stochastic_depth_prob = stochastic_depth_prob

    def __call__(
        self, channels_in: int, channels_out: int, downsample: int = 1
    ) -> nn.Module:
        """
        Create ConvNeXt block.

        Args:
            channels_in: The number of channels in the input.
            channels_out: The number of channels used in the remaining
                layers in the block.
            downsample: Degree of downsampling to be performed. If <= 1
                no downsampling is performed

        Return:
            The ConvNeXt block.
        """
        blocks = []
        if downsample:
            blocks += [
                self.layer_norm_with_permute(channels_in),
                nn.Conv2d(
                    channels_in,
                    channels_out,
                    stride=downsample,
                    kernel_size=downsample
                ),
            ]
        elif channels_in != channels_out:
            blocks += [nn.Conv2d(channels_in, channels_out, stride=1, kernel_size=1)]
        blocks.append(
            models.convnext.CNBlock(
                channels_out,
                self.layer_scale,
                self.stochastic_depth_prob,
                self.norm_factory,
            )
        )
        return nn.Sequential(*blocks)
