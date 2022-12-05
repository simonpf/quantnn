"""
quantnn..models.pytorch.aggregators
===================================

This module provides 'torch.nn.module's that merge two or more input tensors
to produce a single output tensor. They can be used to merge separate
 branches (data streams) in neural networks.
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
        self.aggregator = aggregator_factory(channels_in, 2, channels_in)

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

        x_2_comb, x_1_comb = x_2.intersection(x_1)

        # No merge required, if there are no streams with complementary
        # information.
        if x_2_comb is None:
            if return_full:
                no_merge = no_merge.tensor
            return no_merge

        merged = PackedTensor(
            self.aggregator(x_1_comb.tensor, x_2_comb.tensor),
            x_1_comb.batch_size,
            x_1_comb.batch_indices
        )
        if no_merge is None:
            if return_full:
                merged = merged.tensor
            return merged

        merged = merged.sum(no_merge)
        if return_full:
            return merged.tensor
        return merged


class SparseAggregatorFactory:
    """
    Factory class to create sparse aggregation block.
    """
    def __init__(
            self,
            block_factory
    ):
        """
        Args:
            block_factory: A factory to create the underlying merge block.
        """
        self.block_factory = block_factory

    def __call__(
            self,
            input_channels,
            n_inputs,
            output_channels
    ):
        """
        Create an aggregation block for a given number of input channels.

        Args:
            input_channels: The number of channels in the inputs to merge.
            n_inputs: The number of inputs. Must be 2.
            output_channels: The number of output channels. Must be the same
                as the number of input channels.
        """
        if not n_inputs == 2:
            raise ValueError(
                "The SparseAggregator only support aggregation of two input"
                " streams."
            )
        if not input_channels == output_channels:
            raise ValueError(
                "The 'input_channels' and 'output_channels' must be the same "
                "for SparseAggregator."
            )
        return SparseAggregator(input_channels, self.block_factory)


class AverageBlock(nn.Module):
    """
    Calculates the average of two inputs.
    """
    def forward(self, *args):
        """Average input 'x_1' and 'x_2'."""
        n = len(args)
        x_1 = args[0]
        x_2 = args[1]
        result = torch.add(x_1, x_2)
        for x in args[2:]:
            result = torch.add(result, x)
        return torch.mul(result, 1 / n)


class AverageAggregatorFactory:
    """Aggregator factory for an average aggregator."""

    def __call__(self, *args):
        return AverageBlock()


class SumBlock(nn.Module):
    """
    Calculates the sum of two inputs.
    """

    def forward(self, *args):
        x_1 = args[0]
        x_2 = args[1]
        result = torch.add(x_1, x_2)
        for x in args[2:]:
            result = torch.add(result, x)
        return result


class SumAggregatorFactory:
    """Aggregator factory for a sum aggregator."""

    def __call__(self, *args):
        return SumBlock()


class ConcatenateBlock(nn.Module):
    """
    Applies a single block to the concatenation of the two
    inputs.
    """

    def __init__(self, block, residual=True):
        """
        Args:
            block: A 'nn.Module' to apply to the concatenated inputs.
            residual: If ``True``, ``block`` will be used to calculate
                a residual that is added to the first input stream.
        """
        super().__init__()
        self.block = block
        if isinstance(residual, bool):
            if not residual:
                self.residual = None
            if residual:
                self.residual = 0
        else:
            self.residual = residual

    def forward(self, *args):
        """
        Concatenates inputs and applies the block.
        """
        y = self.block(torch.cat(args, dim=1))
        if self.residual is not None:
            return y + args[self.residual]
        return y


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

    def __call__(
            self,
            channels_in,
            n_inputs,
            channels_out=None,
            residual=True
    ):
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
        block = self.block_factory(n_inputs * channels_in, channels_out)
        return ConcatenateBlock(block, residual=residual)


class LinearAggregatorFactory:
    """
    Aggregation using a pixel-wise, affine transformation applied to the
    concatenated input channels.
    """
    def __init__(
        self,
        norm_factory=None,
    ):
        """
        Args:
            norm_factory:
        """
        if norm_factory is None:
            norm_factory = nn.BatchNorm2d
        self.norm_factory = norm_factory

    def __call__(
            self,
            channels_in,
            n_inputs,
            channels_out=None,
            residual=True
    ):
        if channels_out is None:
            channels_out = channels_in
        if self.norm_factory is not None:
            return ConcatenateBlock(
                nn.Sequential(
                    nn.Conv2d(n_inputs * channels_in, channels_out, kernel_size=1),
                    self.norm_factory(channels_out),
                ),
                residual=residual
            )
        return ConcatenateBlock(
            nn.Conv2d(n_inputs * channels_in, channels_out, kernel_size=1),
            residual=residual
        )
