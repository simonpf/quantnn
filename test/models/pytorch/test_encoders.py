"""
Tests for quantnn.models.pytorch.encoders
"""
import torch

from quantnn.models.pytorch.torchvision import ResNetBlockFactory
from quantnn.models.pytorch.encoders import (
    SpatialEncoder,
)


def test_spatial_encoder():
    """
    Test propagation through a spatial encoder and make sure that
    dimensions are changed as expected.
    """
    block_factory = ResNetBlockFactory()
    stages = [2, 2, 2, 2]
    encoder = SpatialEncoder(
        input_channels=1,
        stages=stages,
        block_factory=block_factory,
        channel_scaling=2,
        max_channels=8
    )
    # Test forward without skip connections.
    x = torch.ones((1, 1, 32, 32))
    y = encoder(x)
    # Width and height should be reduced by 16.
    # Number of channels should be maximum.
    assert y.shape == (1, 8, 2, 2)

    # Test forward width skips returned.
    y = encoder(x, return_skips=True)
    # Number of outputs is number of stages + 1.
    assert len(y) == 5
    # First element is just the input.
    assert y[0].shape == (1, 1, 32, 32)
    # First element is output from last layer.
    assert y[-1].shape == (1, 8, 2, 2)
