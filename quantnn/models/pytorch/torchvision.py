"""
quantnn.models.pytorch.torchvision
==================================

Wrapper for models defined in torchvision models.

Using this module obviously requires torchvision to be installed.
"""
from copy import copy
from functools import partial
from typing import Optional, Callable, Tuple, Union, List

import torch
from torch import nn
from torchvision import models
from torchvision.ops import Permute
from torchvision.models import swin_transformer


from quantnn.models.pytorch.downsampling import PatchMergingBlock
from quantnn.models.pytorch.normalization import LayerNormFirst


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
        self,
        channels_in: int,
        channels_out: int,
        downsample: Optional[Union[int, Tuple[int]]] = None,
        block_index: int = 0,
    ) -> nn.Module:
        """
        Create ResNet block.

        Args:
            channels_in: The number of channels in the input.
            channels_out: The number of channels used in the remaining
                layers in the block.
            downsample: Degree of downsampling to be performed by the block.
                No downsampling is performed if None.

        Return:
            The ResNet block.
        """
        stride = (1, 1)
        projection = None

        if isinstance(downsample, int):
            downsample = (downsample,) * 2

        if downsample is not None and max(downsample) > 1:
            stride = downsample
        if max(stride) > 1 or channels_in != channels_out:
            projection = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, stride=stride, kernel_size=1),
                self.norm_factory(channels_out),
            )

        return models.resnet.BasicBlock(
            channels_in,
            channels_out,
            stride=stride,
            downsample=projection,
            norm_layer=self.norm_factory,
        )


class ResNeXtBlockFactory:
    """
    Factory wrapper for ``torchvision`` ResNeXt blocks.
    """

    def __init__(
        self, cardinality=32, norm_factory: Optional[Callable[[int], nn.Module]] = None
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
        self,
        channels_in: int,
        channels_out: int,
        downsample: Optional[Union[Tuple[int], int]] = 1,
        block_index: int = 0,
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
        stride = (1, 1)
        projection = None

        if isinstance(downsample, int):
            downsample = (downsample,) * 2

        if downsample is not None and max(downsample) > 1:
            stride = downsample

        if max(stride) > 1 or channels_in != channels_out:
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
            norm_layer=self.norm_factory,
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
            Permute((0, 2, 3, 1)), cls.layer_norm(channels_in), Permute((0, 3, 1, 2))
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
        self,
        channels_in: int,
        channels_out: int,
        downsample: Optional[Union[Tuple[int], int]] = None,
        block_index: int = 0,
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
        if downsample is not None:
            blocks += [
                self.layer_norm_with_permute(channels_in),
                nn.Conv2d(
                    channels_in, channels_out, stride=downsample, kernel_size=downsample
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


class SwinBlock(nn.Module):
    """
    Generic wrapper around a swin transfromer block. This wrapper adapts
    the torchvision implementation of the swin transformer block so that
    in can be used within the generic encoder and decoder models of
    'quantnn'.
    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        n_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        version=1,
    ):
        """
        Args:
            channels_in: The number of channel in the block input.
            channels_out: The number of channels within the block and its output.
            n_heads: The number of attention heads to be used.
            window_size: The size of the windows over which the attention is
                computed.
            shift_size: The size of the


        """
        super().__init__()
        if channels_in != channels_out:
            self.proj = nn.Linear(channels_in, channels_out)
        else:
            self.proj = nn.Identity()
        if version == 1:
            self.body = swin_transformer.SwinTransformerBlock(
                dim=channels_out,
                num_heads=n_heads,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                stochastic_depth_prob=stochastic_depth_prob,
                norm_layer=nn.LayerNorm,
                attn_layer=swin_transformer.ShiftedWindowAttention,
            )
        else:
            self.body = swin_transformer.SwinTransformerBlockV2(
                dim=channels_out,
                num_heads=n_heads,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                stochastic_depth_prob=stochastic_depth_prob,
                norm_layer=nn.LayerNorm,
                attn_layer=swin_transformer.ShiftedWindowAttentionV2,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward tensor through block taking into account the different
        channel order assumed by the swin transformer.
        """
        x_t = torch.permute(x, (0, 2, 3, 1))
        y = self.body(self.proj(x_t))
        return torch.permute(y, (0, 3, 1, 2))


class SwinBlockFactory:
    def __init__(
        self,
        window_size: List[int] = [7, 7],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        version=1,
    ):
        """
        Args:
            window_size: The size of the windows over which the attention is
                computed.
            mlp_ratio: The ratio of the channels in the MLP module.
            dropout: Dropout applied in the MLP block.
            attention_dropout: Dropout applied to the computed attention
            stochastic_depth_prob: Probability for stochastic depth.
            version: Which version of swin blocks to apply.
        """
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.stochastic_depth_prob = stochastic_depth_prob
        self.version = version

    def __call__(
        self,
        channels_in,
        channels_out,
        downsample: Optional[Union[int, Tuple[int]]] = None,
        block_index: int = 0,
        n_heads: int = 4,
    ):
        """
        Args:
            channels_in: The number of incoming channels.
            channels_out: The number of outgoing channels.
            downsample: Optional downsampling factors to apply to the input.
            block_index: Optional index informing the factory on which
                the rank of the block in the within the stage.
            n_heads: The number of attnetion heads in the block.

        Return:
            A pytorch module implementing a swin transformer block.
        """
        shift_size = [0, 0]
        if block_index % 2 != 0:
            shift_size = [sze // 2 for sze in self.window_size]

        if downsample is None:
            return SwinBlock(
                channels_in=channels_in,
                channels_out=channels_out,
                n_heads=n_heads,
                window_size=self.window_size,
                shift_size=shift_size,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                attention_dropout=self.attention_dropout,
                stochastic_depth_prob=self.stochastic_depth_prob,
                version=self.version,
            )

        return nn.Sequential(
            PatchMergingBlock(
                channels_in, channels_out, f_dwn=downsample, norm_factory=LayerNormFirst
            ),
            SwinBlock(
                channels_in=channels_out,
                channels_out=channels_out,
                n_heads=n_heads,
                window_size=self.window_size,
                shift_size=shift_size,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                attention_dropout=self.attention_dropout,
                stochastic_depth_prob=self.stochastic_depth_prob,
                version=self.version,
            ),
        )
