"""
Tests for the quantnn.models.pytorch.upsampling module.
"""
import torch
from torch import nn

from quantnn.models.pytorch.upsampling import (
    BilinearFactory,
    UpsampleFactory,
    UpConvolutionFactory,
)


def test_bilinear():
    """
    Ensure bilinear upsampler factory supports upsampling with different
    factors along different dimensions.
    """
    up = BilinearFactory()(channels_in=16, channels_out=8, factor=(2, 4))
    x = torch.rand(2, 16, 32, 32)
    y = up(x)

    assert y.shape[1] == 8
    assert y.shape[-2] == 64
    assert y.shape[-1] == 128


def test_upsample():
    """
    Ensure that generic upsampling factory supports upsampling with different
    factors along different dimensions.
    """
    fac = UpsampleFactory(mode="nearest")
    up = fac(channels_in=16, channels_out=8, factor=(2, 4))
    x = torch.rand(2, 16, 32, 32)
    y = up(x)

    assert y.shape[1] == 8
    assert y.shape[-2] == 64
    assert y.shape[-1] == 128


def test_upconvolution():
    """
    Ensure that the up-convolution factory supports upsampling with different
    factors along different dimensions.
    """
    fac = UpConvolutionFactory(mode="nearest")
    up = fac(channels_in=16, channels_out=8, factor=(2, 4))
    x = torch.rand(2, 16, 32, 32)
    y = up(x)

    assert y.shape[1] == 8
    assert y.shape[-2] == 64
    assert y.shape[-1] == 128
