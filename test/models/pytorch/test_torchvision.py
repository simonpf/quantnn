"""
Tests for the torchvision wrappers defined in
quantnn.models.pytorch.torchvision.
"""
import pytest
import torch

import quantnn.models.pytorch.torchvision as tv
try:
    import quantnn.models.pytorch.torchvision as tv

    HAS_TORCHVISION = True
except ImportError as e:
    HAS_TORCHVISION = False


@pytest.mark.skipif(not HAS_TORCHVISION, reason="torchvision not available")
def test_resnet_block():
    """
    Ensure that the ResNet factory produces an nn.Module and that
    the output has the specified number of channels.
    """
    x = torch.ones((1, 1, 8, 8))

    factory = tv.ResNetBlockFactory()
    block = factory(1, 2)
    y = block(x)
    assert y.shape == (1, 2, 8, 8)

    block = factory(1, 2, downsample=2)
    y = block(x)
    assert y.shape == (1, 2, 4, 4)

    block = factory(1, 2, downsample=(1, 2))
    y = block(x)
    assert y.shape == (1, 2, 8, 4)


@pytest.mark.skipif(not HAS_TORCHVISION, reason="torchvision not available")
def test_convnext_block():
    """
    Ensure that the ConvNeXt factory produces an nn.Module and that
    the output has the specified number of channels.
    """
    x = torch.ones((1, 1, 8, 8))

    factory = tv.ConvNeXtBlockFactory()
    block = factory(1, 2)
    y = block(x)
    assert y.shape == (1, 2, 8, 8)

    block = factory(1, 2, downsample=2)
    y = block(x)
    assert y.shape == (1, 2, 4, 4)

    block = factory(1, 2, downsample=(1, 2))
    y = block(x)
    assert y.shape == (1, 2, 8, 4)


@pytest.mark.skipif(not HAS_TORCHVISION, reason="torchvision not available")
def test_swin_transformer_block():
    """
    Ensure that the ConvNeXt factory produces an nn.Module and that
    the output has the specified number of channels.
    """
    x = torch.ones((1, 16, 32, 32))

    factory = tv.SwinBlockFactory()
    block = factory(16, 16)
    y = block(x)
    assert y.shape == (1, 16, 32, 32)

    block = factory(16, 32, downsample=2)
    y = block(x)
    assert y.shape == (1, 32, 16, 16)

    block = factory(16, 32, downsample=4)
    y = block(x)
    assert y.shape == (1, 32, 8, 8)
