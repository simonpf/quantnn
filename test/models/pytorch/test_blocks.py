from quantnn.models.pytorch.blocks import (
    ConvBlockFactory,
    ConvTransposedBlockFactory,
    ResNeXtBlockFactory,
    ConvNextBlockFactory
)

import torch
from torch import nn


def test_conv_block_factory():
    """
    Ensures that the basic ConvBlockFactory produces a working
    block.
    """
    block_factory = ConvBlockFactory(kernel_size=3)
    block = block_factory(8, 16)
    input = torch.ones(8, 8, 8, 8)
    output = block(input)
    assert output.shape == (8, 16, 8, 8)

    block_factory = ConvBlockFactory(
        kernel_size=3,
        norm_factory=nn.BatchNorm2d,
        activation_factory=nn.GELU
    )
    block = block_factory(8, 16)
    input = torch.ones(8, 8, 8, 8)
    output = block(input)
    assert output.shape == (8, 16, 8, 8)


def test_conv_transposed_block_factory():
    """
    Ensures that the basic ConvBlockFactory produces a working
    block.
    """
    block_factory = ConvTransposedBlockFactory(kernel_size=3)
    block = block_factory(8, 16)
    input = torch.ones(8, 8, 8, 8)
    output = block(input)
    assert output.shape == (8, 16, 8, 8)

    block_factory = ConvTransposedBlockFactory(
        kernel_size=3,
        norm_factory=nn.BatchNorm2d,
        activation_factory=nn.GELU
    )
    block = block_factory(8, 16)
    input = torch.ones(8, 8, 8, 8)
    output = block(input)
    assert output.shape == (8, 16, 8, 8)


def test_resnext_block():
    """
    Ensure that the ResNext factory produces an nn.Module and that
    the output has the specified number of channels.
    """
    x = torch.ones((1, 1, 8, 8))

    factory = ResNeXtBlockFactory()
    block = factory(1, 64)
    y = block(x)
    assert y.shape == (1, 64, 8, 8)

    block = factory(1, 64, downsample=2)
    y = block(x)
    assert y.shape == (1, 64, 4, 4)


def test_convnext_block():
    """
    Ensure that the ConvNext factory produces an nn.Module and that
    the output has the specified number of channels.
    """
    x = torch.ones((1, 1, 8, 8))

    factory = ConvNextBlockFactory(version=1)
    block = factory(1, 64)
    y = block(x)
    assert y.shape == (1, 64, 8, 8)
    block = factory(1, 64, downsample=2)
    y = block(x)
    assert y.shape == (1, 64, 4, 4)

    factory = ConvNextBlockFactory(version=2)
    block = factory(1, 64)
    y = block(x)
    assert y.shape == (1, 64, 8, 8)
    block = factory(1, 64, downsample=2)
    y = block(x)
    assert y.shape == (1, 64, 4, 4)
