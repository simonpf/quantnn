"""
Tests for quantnn.models.pytorch.encoders
"""
import torch
from torch import nn

from quantnn.models.pytorch import factories
from quantnn.models.pytorch.torchvision import ResNetBlockFactory
from quantnn.models.pytorch.aggregators import AverageAggregatorFactory
from quantnn.models.pytorch.encoders import (
    SpatialEncoder,
    MultiInputSpatialEncoder,
    ParallelEncoderLevel,
    ParallelEncoder,
)


def test_spatial_encoder():
    """
    Test propagation through a spatial encoder and make sure that
    dimensions are changed as expected.
    """
    block_factory = ResNetBlockFactory()
    stages = [2, 2, 2, 2]
    encoder = SpatialEncoder(
        channels=1,
        stages=stages,
        block_factory=block_factory,
        channel_scaling=2,
        max_channels=4
    )
    # Test forward without skip connections.
    x = torch.ones((1, 1, 32, 32))
    y = encoder(x)
    # Width and height should be reduced by 8.
    # Number of channels should be maximum.
    assert y.shape == (1, 4, 4, 4)

    # Test forward width skips returned.
    y = encoder(x, return_skips=True)
    # Number of outputs is number of stages + 1.
    assert len(y) == 4
    # First element is just the input.
    assert y[0].shape == (1, 1, 32, 32)
    # Last element is output from last layer.
    assert y[-1].shape == (1, 4, 4, 4)

    # Repeat tests with explicitly specified channel numbers.
    encoder = SpatialEncoder(
        channels=[1, 8, 4, 2],
        stages=stages,
        block_factory=block_factory,
        channel_scaling=2,
        max_channels=8
    )
    x = torch.ones((1, 1, 32, 32))
    y = encoder(x)
    assert y.shape == (1, 2, 4, 4)
    y = encoder(x, return_skips=True)
    assert len(y) == 4
    assert y[0].shape == (1, 1, 32, 32)
    assert y[1].shape == (1, 8, 16, 16)
    assert y[2].shape == (1, 4, 8, 8)
    assert y[3].shape == (1, 2, 4, 4)

    # Repeat tests with different downscaling factors.
    encoder = SpatialEncoder(
        channels=[1, 8, 4, 2],
        stages=stages,
        block_factory=block_factory,
        channel_scaling=2,
        max_channels=8,
        downsampling_factors=[2, 3, 4]
    )
    x = torch.ones((1, 1, 96, 96))
    y = encoder(x)
    assert y.shape == (1, 2, 4, 4)
    y = encoder(x, return_skips=True)
    assert len(y) == 4
    assert y[0].shape == (1, 1, 96, 96)
    assert y[1].shape == (1, 8, 48, 48)
    assert y[2].shape == (1, 4, 16, 16)
    assert y[3].shape == (1, 2, 4, 4)


def test_spatial_encoder_downsampler_factory():
    """
    Test propagation through a spatial encoder with an explicit
    downsampling factory.
    """
    block_factory = ResNetBlockFactory()
    downsampler_factory = factories.MaxPooling()
    stages = [2, 2, 2, 2]
    encoder = SpatialEncoder(
        channels=1,
        stages=stages,
        block_factory=block_factory,
        channel_scaling=2,
        max_channels=4,
        downsampler_factory=downsampler_factory
    )
    # Test forward without skip connections.
    x = torch.ones((1, 1, 32, 32))
    y = encoder(x)
    # Width and height should be reduced by 8.
    # Number of channels should be maximum.
    assert y.shape == (1, 4, 4, 4)


def test_multi_input_spatial_encoder():
    """
    Test propagation through a spatial encoder and make sure that
    dimensions are changed as expected.
    """
    block_factory = ResNetBlockFactory()
    aggregator_factory = AverageAggregatorFactory()
    stages = [2, 2, 2, 2]
    inputs = {
        "input_0": (0, 12),
        "input_1": (2, 8),
        "input_2": (3, 4)
    }
    encoder = MultiInputSpatialEncoder(
        inputs=inputs,
        channels=1,
        stages=stages,
        block_factory=block_factory,
        aggregator_factory=aggregator_factory,
        channel_scaling=2,
        max_channels=8
    )
    # Test forward without skip connections.
    x = {
        "input_0": torch.ones((1, 12, 32, 32)),
        "input_1": torch.ones((1, 8, 16, 16)),
        "input_2": torch.ones((1, 4, 8, 8)),
    }
    y = encoder(x)
    # Width and height should be reduced by 8.
    # Number of channels should be maximum.
    assert y.shape == (1, 8, 4, 4)

    # Test forward width skips returned.
    y = encoder(x, return_skips=True)
    # Number of outputs is number of stages + 1.
    assert len(y) == 4
    # First element is not downsampled.
    assert y[0].shape == (1, 1, 32, 32)
    # First element is output from last layer.
    assert y[-1].shape == (1, 8, 4, 4)


def test_zero_depth_stage():
    """
    Ensure that zero-depth stages are supported.
    """
    block_factory = ResNetBlockFactory()
    aggregator_factory = AverageAggregatorFactory()
    stages = [0, 2, 2, 2]
    inputs = {
        "input_0": (0, 12),
        "input_1": (2, 8),
        "input_2": (3, 4)
    }
    encoder = MultiInputSpatialEncoder(
        inputs=inputs,
        channels=1,
        stages=stages,
        block_factory=block_factory,
        aggregator_factory=aggregator_factory,
        channel_scaling=2,
        max_channels=8
    )
    # Test forward without skip connections.
    x = {
        "input_0": torch.ones((1, 12, 32, 32)),
        "input_1": torch.ones((1, 8, 16, 16)),
        "input_2": torch.ones((1, 4, 8, 8)),
    }
    y = encoder(x)
    # Width and height should be reduced by 8.
    # Number of channels should be maximum.
    assert y.shape == (1, 8, 4, 4)

    # Test forward width skips returned.
    y = encoder(x, return_skips=True)
    # Number of outputs is number of stages + 1.
    assert len(y) == 4
    # First element is not downsampled.
    assert y[0].shape == (1, 1, 32, 32)
    # First element is output from last layer.
    assert y[-1].shape == (1, 8, 4, 4)


def test_spatial_encoder_w_stem():
    """
    Test propagation through a spatial encoder with a stem.
    """
    block_factory = ResNetBlockFactory()
    stages = [2, 2, 2, 2]
    stem_factory = lambda n_chans: block_factory(13, n_chans, 3)
    encoder = SpatialEncoder(
        channels=1,
        stages=stages,
        block_factory=block_factory,
        channel_scaling=2,
        max_channels=8,
        stem_factory=stem_factory
    )
    # Test forward without skip connections.
    x = torch.ones((1, 13, 3 * 32, 3 * 32))
    y = encoder(x)
    # Width and height should be reduced by 16.
    # Number of channels should be maximum.
    assert y.shape == (1, 8, 4, 4)

    # Test forward width skips returned.
    y = encoder(x, return_skips=True)
    # Number of outputs is number of stages + 1.
    assert len(y) == 4
    # First element is output from first layer.
    assert y[0].shape == (1, 1, 32, 32)
    # First element is output from last layer.
    assert y[-1].shape == (1, 8, 4, 4)

    # Repeat tests with explicitly specified channel numbers.
    encoder = SpatialEncoder(
        channels=[1, 8, 4, 2],
        stages=stages,
        block_factory=block_factory,
        channel_scaling=2,
        max_channels=8,
        stem_factory=stem_factory
    )
    x = torch.ones((1, 13, 3 * 32, 3 * 32))
    y = encoder(x)
    assert y.shape == (1, 2, 4, 4)
    y = encoder(x, return_skips=True)
    assert len(y) == 4
    assert y[0].shape == (1, 1, 32, 32)
    assert y[1].shape == (1, 8, 16, 16)
    assert y[2].shape == (1, 4, 8, 8)
    assert y[3].shape == (1, 2, 4, 4)

    # Repeat tests with different downscaling factors.
    encoder = SpatialEncoder(
        channels=[1, 8, 4, 2],
        stages=stages,
        block_factory=block_factory,
        channel_scaling=2,
        max_channels=8,
        downsampling_factors=[2, 3, 4]
    )
    x = torch.ones((1, 1, 96, 96))
    y = encoder(x)
    assert y.shape == (1, 2, 4, 4)
    y = encoder(x, return_skips=True)
    assert len(y) == 4
    assert y[0].shape == (1, 1, 96, 96)
    assert y[1].shape == (1, 8, 48, 48)
    assert y[2].shape == (1, 4, 16, 16)
    assert y[3].shape == (1, 2, 4, 4)


def test_parallel_encoder_stage():
    """
    Test propagation through a single parallel encoder stage.
    """
    block_factory = ResNetBlockFactory()
    aggregator_factory = AverageAggregatorFactory()
    def downsampler_factory(ch_in, ch_out, factor):
        return nn.AvgPool2d(kernel_size=factor, stride=factor)

    channels = [2, 4, 8, 16]
    scales = [1, 4, 16, 32]

    encoder = ParallelEncoderLevel(
        channels=channels,
        scales=scales,
        level_index=2,
        depth=4,
        block_factory=block_factory,
        downsampler_factory=downsampler_factory,
        aggregator_factory=aggregator_factory
    )

    x = [
        torch.ones((2, 2, 64, 64)),
        torch.ones((2, 4, 16, 16)),
    ]
    y = encoder(x)

    assert len(y) == 3
    assert y[0].shape == (2, 2, 64, 64)
    assert y[1].shape == (2, 4, 16, 16)
    assert y[2].shape == (2, 8, 4, 4)


def test_parallel_encoder():
    """
    Test propagation through a whole parallel encoder.
    """
    block_factory = ResNetBlockFactory()
    aggregator_factory = AverageAggregatorFactory()
    def downsampler_factory(ch_in, ch_out, factor):
        return nn.AvgPool2d(kernel_size=factor, stride=factor)

    channels = [2, 4, 8]
    scales = [1, 4, 8]

    encoder = ParallelEncoder(
        inputs = {
            0: 4,
            1: 4,
        },
        channels=channels,
        scales=scales,
        depth=4,
        block_factory=block_factory,
        downsampler_factory=downsampler_factory,
        aggregator_factory=aggregator_factory,
        input_aggregator_factory=aggregator_factory,
    )

    x = [
        torch.ones((2, 4, 128, 128)),
        torch.ones((2, 4, 32, 32)),
    ]
    y = encoder(x)

    assert len(y) == 3
    assert y[0].shape == (2, 2, 128, 128)
    assert y[1].shape == (2, 4, 32, 32)
    assert y[2].shape == (2, 8, 16, 16)
