"""
quantnn.models.pytorch.stages
=============================

Implements generic stages used in back-bone models.
"""
import numpy as np
import torch
from torch import nn

from quantnn.models.pytorch import blocks


class AggregationTreeNode(nn.Module):
    """
    Represents a node in an aggregation tree.

    Aggregation in all right child nodes are combined so that
    aggregation is in principle only performed in the nodes at
    the lowest level.
    """

    def __init__(
        self,
        channels_in,
        channels_out,
        level,
        block_factory,
        aggregator_factory,
        channels_agg=0,
        downsample=1,
        block_args=None,
        block_kwargs=None,
    ):
        super().__init__()

        if block_args is None:
            block_args = []
        if block_kwargs is None:
            block_kwargs = {}

        if channels_agg == 0:
            channels_agg = 2 * channels_out

        self.aggregator = None
        self.level = level
        if level <= 0:
            self.left = block_factory(
                channels_in,
                channels_out,
                *block_args,
                downsample=downsample,
                **block_kwargs,
            )
            self.right = None
        elif level == 1:
            self.aggregator = aggregator_factory(channels_agg, channels_out)
            self.left = block_factory(
                channels_in,
                channels_out,
                *block_args,
                downsample=downsample,
                **block_kwargs,
            )
            self.right = block_factory(
                channels_out,
                channels_out,
                *block_args,
                downsample=1,
                **block_kwargs,
            )
        else:
            self.aggregator = None
            self.left = AggregationTreeNode(
                channels_in,
                channels_out,
                level - 1,
                block_factory,
                aggregator_factory,
                downsample=downsample,
                block_args=block_args,
                block_kwargs=block_kwargs,
            )
            self.right = AggregationTreeNode(
                channels_out,
                channels_out,
                level - 1,
                block_factory,
                aggregator_factory,
                channels_agg=channels_agg + channels_out,
                downsample=1,
                block_args=block_args,
                block_kwargs=block_kwargs,
            )

    def forward(self, x, pass_through=None):
        """
        Forward input through tree and aggregate results from child nodes.
        """
        if self.level <= 0:
            return self.left(x)

        if pass_through is None:
            pass_through = []

        y_1 = self.left(x)
        if self.aggregator is not None:
            y_2 = self.right(y_1)
            pass_through = pass_through + [y_1, y_2]
            return self.aggregator(torch.cat(pass_through, 1))

        return self.right(y_1, pass_through + [y_1])


class AggregationTreeRoot(AggregationTreeNode):
    """
    Root of an aggregation tree.
    """

    def __init__(
        self,
        channels_in,
        channels_out,
        tree_height,
        block_factory,
        aggregator_factory,
        downsample=1,
        block_args=None,
        block_kwargs=None,
    ):
        channels_agg = 2 * channels_out + channels_in
        super().__init__(
            channels_in,
            channels_out,
            tree_height,
            block_factory,
            aggregator_factory,
            channels_agg=channels_agg,
            downsample=1,
        )

        self.downsampler = None
        if downsample > 1:
            self.downsampler = nn.MaxPool2d(kernel_size=downsample)

    def forward(self, x):
        """
        Forward input through tree and aggregate results from child nodes.
        """
        if self.level == 0:
            return self.left(x)

        if self.downsampler is not None:
            x = self.downsampler(x)

        pass_through = []

        y_1 = self.left(x)
        if self.aggregator is not None:
            y_2 = self.right(y_1)
            pass_through = pass_through + [y_1, y_2, x]
            return self.aggregator(torch.cat(pass_through, 1))

        return self.right(y_1, pass_through + [y_1, x])


class AggregationTreeFactory:
    """
    An aggregation tree implementing hierarchical aggregation of blocks in a stage.
    """

    def __init__(self, aggregator_factory=None):
        if aggregator_factory is None:
            aggregator_factory = blocks.ConvBlockFactory(
                norm_factory=nn.BatchNorm2d, activation_factory=nn.ReLU
            )
        self.aggregator_factory = aggregator_factory

    def __call__(
        self,
        channels_in,
        channels_out,
        n_blocks,
        block_factory,
        downsample=1,
        block_args=None,
        block_kwargs=None,
    ):
        n_levels = np.log2(n_blocks)
        return AggregationTreeRoot(
            channels_in,
            channels_out,
            n_levels,
            block_factory,
            self.aggregator_factory,
            downsample=downsample,
            block_args=None,
            block_kwargs=None,
        )
