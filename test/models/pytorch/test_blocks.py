from quantnn.models.pytorch.blocks import ConvBlockFactory

import torch
from torch import nn


def test_conv_block_factory():
    """
    Ensures that the basic ConvBlockFactory produces a working
    block.
    """
    block_factory = ConvBlockFactory()
    block = block_factory(8, 16)
    input = torch.ones(8, 8, 8, 8)
    output = block(input)
    assert output.shape == (8, 16, 8, 8)

    block_factory = ConvBlockFactory(
        norm_factory=nn.BatchNorm2d,
        activation_factory=nn.GELU
    )
    block = block_factory(8, 16)
    input = torch.ones(8, 8, 8, 8)
    output = block(input)
    assert output.shape == (8, 16, 8, 8)
