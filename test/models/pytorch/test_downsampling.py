"""
Tests for the quantnn.models.pytorch.downsampling module.
"""
import torch
from torch import nn

from quantnn.models.pytorch.downsampling import ConvNextDownsamplerFactory


def test_convnext_downsampling():
    """
    Test that ConvNext downsampling works with different downsampling factors
    along dimensions.
    """
    down = ConvNextDownsamplerFactory()(16, 16, (2, 4))
    x = torch.rand(2, 16, 32, 32)
    y = down(x)

    assert y.shape[-2] == 16
    assert y.shape[-1] == 8
