"""
quantnn..models.pytorch.aggregators
===================================

This module provides 'torch.nn.modules' that merge two input tensors
to produce a single output tensor. They can be used to merge
 separate branches (data streams) in neural networks.
"""
import torch
from torch import nn
from quantnn.packed_tensor import PackedTensor


class SparseAggregator(nn.Module):
    """
    Aggregator for packed tensors.

    This aggregator combines inputs with missing samples represented
    using PackedTensors. Samples that are present in only one of the
    inputs are directly forwarded to the output while samples that
    are present in both inputs are merged using the provided aggregator
    block.

    The aggregator combines to inputs with the same number of channels
    into a single output with the same number of channels.
    """

    def __init__(self, channels_in, aggregator_factory):
        """
        Args:
            channels_in: The number of channels in each data stream and
                the number of channels in the output of the aggregator.
            aggregator_factory: Aggregator factory to create the module
                that is used to merge the samples that are present in
                both streams.
        """
        super().__init__()
        self.channels_in = channels_in
        self.aggregator = aggregator_factory(2 * channels_in, channels_in)

    def forward(self, x_1, x_2):
        """
        Combined inputs 'x_1' and 'x_2' into a single output with the
        same number of channels.

        Args:
            x_1: A standard 'torch.Tensor' or a 'PackedTensor' containing
                the input from the first data stream.
            x_2: A standard 'torch.Tensor' or a 'PackedTensor' containing
                the input from the second data stream.

        Return:
            If both inputs are dense 'torch.Tensor' objects, the output is
            just the output from the underlying aggregator block. If one
            or both  inputs have missing samples the result is a
            'PackedTensor' that contains the inputs samples that are present
            in only one of the inputs and the result of the aggregator block
            applied to all samples that are present in both inputs.
        """
        # No missing samples in batch.
        if (not isinstance(x_1, PackedTensor)) and (not isinstance(x_2, PackedTensor)):
            return self.aggregator(x_1, x_2)

        return_full = False

        if not isinstance(x_1, PackedTensor):
            batch_size = x_1.shape[0]
            x_1 = PackedTensor(x_1, batch_size, range(batch_size))
            return_full = True

        if not isinstance(x_2, PackedTensor):
            batch_size = x_2.shape[0]
            x_2 = PackedTensor(x_2, batch_size, range(batch_size))
            return_full = True

        try:
            no_merge = x_2.difference(x_1)
        except IndexError:
            # Both streams are empty, do nothing.
            if return_full:
                x_1 = x_1.tensor
            return x_1

        x_1_comb, x_2_comb = x_2.intersection(x_1)

        # No merge required, if there are no streams with complementary
        # information.
        if x_2_comb is None:
            if return_full:
                no_merge = no_merge.tensor
            return no_merge

        merged = self.aggregator(x_1_comb, x_2_comb)
        if no_merge is None:
            if return_full:
                merged = merged.tensor
            return merged

        merged = merged.sum(no_merge)
        if return_full:
            return merged.tensor
        return merged


class AverageBlock(nn.Module):
    """
    Calculates the average of two inputs.
    """

    def forward(self, x_1, x_2):
        """Average input 'x_1' and 'x_2'."""
        return torch.mul(torch.add(x_1, x_2), 0.5)


class AverageAggregatorFactory:
    """Aggregator factory for an average aggregator."""

    def __call__(self, *args):
        return AverageBlock()


class SumBlock(nn.Module):
    """
    Calculates the sum of two inputs.
    """

    def forward(self, x_1, x_2):
        return torch.add(x_1, x_2)


class SumAggregatorFactory:
    """Aggregator factory for a sum aggregator."""

    def __call__(self, *args):
        return SumBlock()


class ConcatenateBlock(nn.Module):
    """
    Applies a single block to the concatenation of the two
    inputs.
    """

    def __init__(self, block):
        """
        Args:
            block: A 'nn.Module' to apply to the concatenated inputs.
        """
        super().__init__()
        self.block = block

    def forward(self, x_1, x_2):
        """
        Concatenates inputs 'x_1' and 'x_2' and applies the block.
        """
        return self.block(torch.cat([x_1, x_2], dim=1))


class BlockAggregatorFactory:
    """
    Generic block aggregator factory.

    The returned aggregator block combines concatenation along channels
    with a block from the objects block factory.
    """

    def __init__(self, block_factory):
        """
        Args: The block factory to use to produce the block applied to
             the concatenated inputs.
        """
        self.block_factory = block_factory

    def __call__(self, channels_in, channels_out=None):
        """
        Create an aggregator block to fuse two inputs with 'channels_in'
        channels and produce output with 'channels_out' channels.

        Args:
            channels_in: The number of channels in each input tensor to
                the produced aggregator block.
            channels_out: The number of channels in the output of the
                produced aggregator block.
        """
        if channels_out is None:
            channels_out = channels_in
        block = self.block_factory(2 * channels_in, channels_out)
        return ConcatenateBlock(block)


class LinearAggregatorFactory:
    """
    Aggregation using a pixel-wise, affine transformation applied to the
    concatenated input channels.
    """

    def __init__(
        self,
        norm_layer=None,
    ):
        """
        Args:
            norm_layer:
        """
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

    def __call__(self, channels_in, channels_out=None):
        if channels_out is None:
            channels_out = channels_in
        if self.norm_layer is not None:
            return ConcatenateBlock(
                nn.Sequential(
                    nn.Conv2d(2 * channels_in, channels_out, kernel_size=1),
                    self.norm_layer(channels_out),
                )
            )
        return ConcatenateBlock(
            nn.Conv2d(2 * channels_in, channels_out, kernel_size=1),
        )
