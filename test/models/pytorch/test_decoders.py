"""
Tests for quantnn.models.pytorch.decoders.
"""
import pytest

import torch
from quantnn.packed_tensor import PackedTensor
from quantnn.models.pytorch.encoders import (
    SpatialEncoder,
    MultiInputSpatialEncoder
)
from quantnn.models.pytorch.decoders import (
    SpatialDecoder,
    SparseSpatialDecoder,
    DLADecoderStage,
    DLADecoder
)
from quantnn.models.pytorch.blocks import ConvBlockFactory
from quantnn.models.pytorch.torchvision import ResNetBlockFactory
from quantnn.models.pytorch.aggregators import (
    LinearAggregatorFactory,
    SparseAggregatorFactory,
    BlockAggregatorFactory
)
from quantnn.models.pytorch.upsampling import BilinearFactory


def test_spatial_decoder():
    """
    Test that chaining an encoder and corresponding decoder reproduces
    output of the same spatial dimensions as the input.
    """
    block_factory = ResNetBlockFactory()
    encoder = SpatialEncoder(
        channels=[1, 2, 4, 8],
        stage_depths=[4] * 4,
        block_factory=block_factory,
    )
    decoder = SpatialDecoder(
        channels=[8, 4, 2, 1],
        stage_depths=[4] * 3,
        block_factory=block_factory,
        skip_connections=False
    )
    # Test forward without skip connections.
    x = torch.ones((1, 1, 32, 32))
    y = decoder(encoder(x))

    # Shape of y should be same as before.
    assert y.shape == (1, 1, 32, 32)

    #
    # Test asymmetric decoder
    #
    decoder = SpatialDecoder(
        channels=[8, 4, 1],
        stage_depths=[4] * 2,
        block_factory=block_factory,
        skip_connections=encoder.skip_connections,
    )
    # Test forward width skips returned.
    y = decoder(encoder(x, return_skips=True))
    # Size should be less than input.
    assert y.shape == (1, 1, 16, 16)

    encoder = SpatialEncoder(
        channels=[1, 2, 16, 32],
        stage_depths=[4] * 4,
        block_factory=block_factory,
        downsampling_factors=[2, 2, 4]
    )
    decoder = SpatialDecoder(
        channels=[32, 16, 2, 1],
        stage_depths=[4] * 3,
        block_factory=block_factory,
        skip_connections=encoder.skip_connections,
        upsampling_factors=[4, 2, 2],
    )
    # Test forward without skip connections.
    x = torch.ones((1, 1, 128, 128))
    y = decoder(encoder(x, return_skips=True))
    # Width and height should be reduced by 16.
    # Number of channels should be maximum.
    assert y.shape == (1, 1, 128, 128)

    # Test decoder with different channel config.
    decoder = SpatialDecoder(
        channels=[32, 2, 2, 2],
        stage_depths=[4] * 3,
        block_factory=block_factory,
        skip_connections=encoder.skip_connections,
        upsampling_factors=[4, 2, 2],
    )
    # Test forward without skip connections.
    x = torch.ones((1, 1, 128, 128))
    y = decoder(encoder(x, return_skips=True))
    # Width and height should be reduced by 16.
    # Number of channels should be maximum.
    assert y.shape == (1, 2, 128, 128)


def test_encoder_decoder_multi_scale_output():
    """
    Tests the chaining of a multi-input encoder and a decoder with
    potentially missing input.
    """
    block_factory = ConvBlockFactory()
    aggregator_factory = SparseAggregatorFactory(
        LinearAggregatorFactory()
    )
    inputs = {
        "input_1": 2,
        "input_2": 4
    }
    encoder = MultiInputSpatialEncoder(
        inputs=inputs,
        channels=[4, 8, 16, 16],
        stage_depths=[4] * 4,
        block_factory=block_factory,
        aggregator_factory=aggregator_factory
    )

    decoder = SparseSpatialDecoder(
        channels=[16, 16, 8, 4],
        stage_depths=[4] * 3,
        block_factory=block_factory,
        skip_connections=encoder.skip_connections,
        multi_scale_output=16
    )
    # Test forward without skip connections.
    x = {
        "input_1": PackedTensor(
            torch.ones((0, 8, 16, 16)),
            batch_size=4,
            batch_indices=[]
        ),
        "input_2": PackedTensor(
            torch.ones((1, 16, 8, 8)),
            batch_size=4,
            batch_indices=[0]
        )
    }
    y = encoder(x, return_skips=True)
    y = decoder(y)
    assert len(y) == 4
    for scale, tensor in y.items():
        assert tensor.shape[1] == 16
    assert y[1].shape[2] == 32

    #
    # Ensure using different channels than encoder works.
    #

    decoder = SparseSpatialDecoder(
        channels=[16, 8, 8, 8],
        stage_depths=[4] * 3,
        block_factory=block_factory,
        skip_connections=encoder.skip_connections,
        multi_scale_output=16
    )
    x = {
        "input_1": PackedTensor(
            torch.ones((0, 8, 16, 16)),
            batch_size=4,
            batch_indices=[]
        ),
        "input_2": PackedTensor(
            torch.ones((1, 16, 8, 8)),
            batch_size=4,
            batch_indices=[0]
        )
    }
    y = encoder(x, return_skips=True)
    y = decoder(y)
    assert y[1].shape[1] == 16
    assert y[8].shape[1] == 16

    #
    # Test decoer with more stages than encoder.
    #

    decoder = SparseSpatialDecoder(
        channels=[16, 8, 8, 8, 2],
        stage_depths=[4] * 4,
        block_factory=block_factory,
        skip_connections=encoder.skip_connections,
    )
    x = {
        "input_1": PackedTensor(
            torch.ones((0, 8, 16, 16)),
            batch_size=4,
            batch_indices=[]
        ),
        "input_2": PackedTensor(
            torch.ones((1, 16, 8, 8)),
            batch_size=4,
            batch_indices=[0]
        )
    }
    y = encoder(x, return_skips=True)
    y = decoder(y)
    assert y.shape[1] == 2
    assert y.shape[2] == 64


@pytest.mark.xfail
def test_dla_decoder():
    """
    Test implementation of the DLA decoder stages and full decoder.
    """
    x = [
        torch.ones((2, 16, 4, 4)),
        torch.ones((2, 8, 16, 16)),
        torch.ones((2, 4, 32, 32)),
        torch.ones((2, 2, 64, 64)),
    ]

    aggregator_factory = BlockAggregatorFactory(
        ResNetBlockFactory()
    )
    upsampler_factory = BilinearFactory()

    #
    # Single stage
    #

    decoder = DLADecoderStage(
        [16, 8, 4, 2],
        [16, 8, 4, 2],
        [16, 4, 2, 1],
        aggregator_factory,
        upsampler_factory
    )
    y = decoder(x)
    # Output should contain one less tensor than the input.
    assert len(y) == len(x) - 1
    y[0].shape == (2, 4, 32, 32)
    y[1].shape == (2, 8, 16, 16)
    y[2].shape == (2, 16, 4, 4)

    #
    # Full decoder
    #

    decoder = DLADecoder(
        [16, 8, 4, 2],
        [16, 4, 2, 1],
        aggregator_factory,
        upsampler_factory
    )
    y = decoder(x[::-1])
    # Output should be a single tensor
    assert isinstance(y, torch.Tensor)
    assert y.shape == (2, 2, 64, 64)

    #
    # Test sparse input.
    #
    aggregator_factory = SparseAggregatorFactory(
        BlockAggregatorFactory(
            ResNetBlockFactory()
        )
    )
    decoder = DLADecoder(
        [16, 8, 4, 2],
        [16, 4, 2, 1],
        aggregator_factory,
        upsampler_factory
    )

    x = [
        PackedTensor(
            torch.ones((0, 16, 4, 4)),
            batch_size=2,
            batch_indices=[]
        ),
        torch.ones((2, 8, 16, 16)),
        torch.ones((2, 4, 32, 32)),
        PackedTensor(
            torch.ones((0, 16, 64, 64)),
            batch_size=2,
            batch_indices=[]
        )
    ]
    y = decoder(x[::-1])

    # Output is, again, a full tensor.
    assert isinstance(y, torch.Tensor)
    assert y.shape == (2, 2, 64, 64)

