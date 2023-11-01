"""
quantnn..models.pytorch.aggregators
===================================

This module provides 'torch.nn.module's that merge two or more input tensors
to produce a single output tensor. They can be used to merge separate
 branches (data streams) in neural networks.
"""
from typing import Callable, Tuple, Union, Optional

import torch
from torch import nn
from torch.nn.functional import softmax

from quantnn.packed_tensor import PackedTensor, forward
import quantnn.models.pytorch.masked as nm


class SparseAggregator(nn.Module):
    """
    Aggregator with support for packed tensors.

    This aggregator combines inputs with missing samples that are
    represented using PackedTensors. Samples that are present in only
    one of the inputs are directly forwarded to the output while
    samples that are present in both inputs are merged using the
    provided aggregator block.
    """
    def __init__(
            self,
            channels_in: Tuple[int],
            channels_out: int,
            aggregator_factory: Callable[[int, int], nn.Module]
    ) -> None:
        """
        Args:
            channels_in: A tupple
            aggregator_factory: Aggregator factory to create the module
                that is used to merge the samples that are present in
                both streams.
        """
        super().__init__()
        projs = []
        aggs = []
        for ch_in in channels_in:
            if hasattr(aggregator_factory, "block_factory"):
                block_factory = aggregator_factory.block_factory
                projs.append(block_factory(ch_in, channels_out))
            else:
                projs.append(nn.Identity())

        agg_in = channels_in[0]
        for ind in range(1, len(channels_in)):
            aggs.append(
                aggregator_factory((agg_in, channels_in[ind]), channels_out)
            )
            agg_in = channels_out

        self.projs = nn.ModuleList(projs)
        self.aggs = nn.ModuleList(aggs)
        self.identity = nn.Identity()

    def _forward(
            self,
            x_1: Union[torch.Tensor, PackedTensor],
            x_2: Union[torch.Tensor, PackedTensor],
            proj_1,
            proj_2,
            agg,
    ) -> Union[torch.Tensor, PackedTensor]:
        """
        Combines inputs 'x_1' and 'x_2' into a single output with the
        same number of channels.

        Args:
            x_1: A standard 'torch.Tensor' or a 'PackedTensor' containing
                the input from the first data stream.
            x_2: A standard 'torch.Tensor' or a 'PackedTensor' containing
                the input from the second data stream.
            proj_1: Projection to apply to x_1, where x_2 is empty.
            proj_2: Projection to apply to x_2, where x_1 is empty.
            agg: Aggregator module to apply aggregate x_1 and x_2.

        Return:
            If both inputs are dense 'torch.Tensor' objects, the output is
            just the output from the underlying aggregator block. If one
            or both  inputs have missing samples the result is a
            'PackedTensor' that contains the inputs samples that are present
            in only one of the inputs and the result of the aggregator block
            applied to all samples that are present in both inputs.
        """
        if x_1 is None:
            if x_2 is not None:
                return proj_2(x_2)
            return x_2
        if x_2 is None:
            if x_1 is not None:
                return proj_1(x_1)
            return x_1

        # No missing samples in batch.
        if (not isinstance(x_1, PackedTensor)) and (not isinstance(x_2, PackedTensor)):
            return agg(x_1, x_2)

        return_full = False

        if not isinstance(x_1, PackedTensor):
            batch_size = x_1.shape[0]
            x_1 = PackedTensor(x_1, batch_size, range(batch_size))
            return_full = True

        if not isinstance(x_2, PackedTensor):
            batch_size = x_2.shape[0]
            x_2 = PackedTensor(x_2, batch_size, range(batch_size))
            return_full = True

        if x_1.empty and x_2.empty:
            # Both streams are empty, do nothing.
            if return_full:
                x_1 = x_1.tensor
            return x_1

        x_1_only, x_2_only, x_1_both, x_2_both = x_1.split_parts(x_2)

        tmpl = x_1.tensor if not x_1.empty else x_2.tensor

        batch_size = x_1.batch_size

        res = None
        if x_1_only is not None:
            res = forward(proj_1, x_1_only)

        if x_2_only is not None:
            x_2_p = forward(proj_2, x_2_only)
            if res is None:
                res = x_2_p
            else:
                res = res.union(x_2_p)
        if x_1_both is not None:
            x_agg = agg(x_1_both.tensor, x_2_both.tensor)
            x_agg = PackedTensor(x_agg, batch_size, x_1_both.batch_indices)
            if res is None:
                res = x_agg
            else:
                res = res.union(x_agg)

        if return_full:
            return res.tensor
        return res

    def forward(
            self,
            x_1: Union[torch.Tensor, PackedTensor],
            *args
    ) -> Union[torch.Tensor, PackedTensor]:
        """
        Recursively combines input tensors into a single output with
        the same number of channels.

        Args:
            x_1: A standard 'torch.Tensor' or a 'PackedTensor' containing
                the input from the first data stream.
            *args: Remaining tensors to combin with x_1.

        Return:
            If all inputs are dense 'torch.Tensor' objects, the output is
            just the output from the recursive application of the underlying
            aggregator blocks.
        """
        proj_1 = self.projs[0]
        for x_2, proj_2, agg in zip(args, self.projs[1:], self.aggs):
            x_1 = self._forward(x_1, x_2, proj_1, proj_2, agg)
            proj_1 = self.identity
        return x_1

class SparseAggregatorFactory:
    """
    Factory class to create sparse aggregation block from a dense aggregation
    block.
    """
    def __init__(
            self,
            agg_factory,
            conditional=True
    ):
        """
        Args:
            block_factory: A factory to create the underlying merge block.
        """
        if conditional:
            def factory(*args, **kwargs):
                return agg_factory(*args, **kwargs)
            self.agg_factory = factory
        else:
            self.agg_factory = agg_factory

    def __call__(
            self,
            channels_in: Tuple[int],
            channels_out: int
    ):
        """
        Create an aggregation block for a given number of input channels.

        Args:
            input_channels: A tuple specifying the number of channels in all
                inputs that should be merged.
            output_channels: The number of output channels. Must be the same
                as the number of input channels.
        """
        return SparseAggregator(channels_in, channels_out, self.agg_factory)


class AverageBlock(nn.Module):
    """
    Calculates the average of two inputs.
    """
    def forward(self, *args):
        """Average input 'x_1' and 'x_2'."""
        n = len(args)
        if n == 0:
            return None
        if n == 1:
            return args[0]
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

    def __init__(self, block, residual=False):
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
        #if self.residual is not None:
        #    res = args[self.residual]
        #    n = min(y.shape[1], res.shape[1])
        #    y[:, :n] += res[:, :n]
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
            channels_out = channels_in[0]
        block = self.block_factory(sum(channels_in), channels_out)
        return ConcatenateBlock(block, residual=residual)


class LinearAggregatorFactory:
    """
    Aggregation using a pixel-wise, affine transformation applied to the
    concatenated input channels.
    """
    def __init__(
        self,
        norm_factory=None,
        masked=False
    ):
        """
        Args:
            norm_factory:
        """
        if masked:
            mod = nm
        else:
            mod = nn

        self.norm_factory = norm_factory

        def block_factory(channels_in, channels_out):
            blocks = [nm.Conv2d(channels_in, channels_out, kernel_size=1)]
            if self.norm_factory is not None:
                blocks.append(norm_factory(channels_out))
            return nn.Sequential(*blocks)
        self.block_factory = block_factory

    def __call__(
            self,
            channels_in,
            channels_out=None,
            residual=True
    ):
        if channels_out is None:
            channels_out = channels_in
        if self.norm_factory is not None:
            return ConcatenateBlock(
                self.block_factory(sum(channels_in), channels_out),
                residual=residual
            )
        return ConcatenateBlock(
            self.block_factory(sum(channels_in), channels_out),
            residual=residual
        )


class AttentionFusion(nn.Module):
    def __init__(
            self,
            channels_in: Tuple[int],
            channels_out: int,
            n_embed: Optional[int] = None,
    ):
        super().__init__()

        if n_embed is None:
            n_embed = channels_out

        self.n_e = n_embed
        self.n_o = channels_out

        self.q = nn.Linear(self.n_e, self.n_o)
        self.k = nn.ModuleList(
            [nn.Linear(chan_in, self.n_e) for chan_in in channels_in]
        )
        self.v = nn.ModuleList(
            [nn.Linear(chan_in, self.n_o) for chan_in in channels_in]
        )

        self.scl = 1.0 / torch.sqrt(torch.tensor(self.n_e))

    def forward(self, *args):

        n_b, _, height, width = args[0].shape
        dtype = args[0].dtype
        x_ts = [
            x.permute(0, 2, 3, 1).view((n_b, height * width, -1))
            for x in args
        ]

        shape = (n_b, height * width, -1)
        k = torch.stack([K(x_t).view(shape) for K, x_t in zip(self.k, x_ts)], 2)
        v = torch.stack([V(x_t).view(shape) for V, x_t in zip(self.v, x_ts)], 2)

        att = softmax(self.scl.to(dtype) * self.q(k), -1)

        out = (att * v).sum(-2).view(n_b, width, height, self.n_o)
        return out.permute((0, 3, 1, 2))
