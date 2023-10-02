"""
quantnn.models.pytorch.downsampling
===================================

This module provides factory classes for downsampling modules.
"""
from typing import Union, Tuple, Callable

import torch
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


class PatchMergingBlock(nn.Module):
    """
    Implements patch merging as employed in the Swin architecture.
    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        f_dwn: Union[int, Tuple[int, int]],
        norm_factory: Callable[[int], nn.Module] = None,
    ):
        super().__init__()
        if isinstance(f_dwn, tuple):
            if f_dwn[0] != f_dwn[1]:
                raise ValueError(
                    "Downsampling by patch merging only supports homogeneous "
                    "downsampling factors."
                )
            f_dwn = f_dwn[0]
        self.f_dwn = f_dwn
        channels_d = channels_in * f_dwn**2

        if norm_factory is None:
            norm_factory = LayerNormFirst

        self.norm = norm_factory(channels_d)

        if channels_d != channels_out:
            self.proj = nn.Conv2d(channels_d, channels_out, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downsample tensor.
        """
        x_d = nn.functional.pixel_unshuffle(x, self.f_dwn)
        return self.proj(self.norm(x_d))


class PatchMergingFactory:
    """
    A factory class to create patch merging downsamplers as employed
    by the swin architecture.
    """

    def __call__(
        self, channels_in: int, channels_out: int, f_dwn: Union[int, Tuple[int, int]]
    ):
        return PatchMergingBlock(channels_in, channels_out, f_dwn)
