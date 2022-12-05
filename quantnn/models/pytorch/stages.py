"""
quantnn.models.pytorch.stages
=============================

Implements generic stages used in back-bone models.
"""
import numpy as np
import torch
from torch import nn


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
            downsample=1
    ):
        super().__init__()

        if channels_agg == 0:
            channels_agg = 2 * channels_out

        self.aggregator = None
        self.level = level
        if level <= 0:
            self.left = block_factory(
                channels_in,
                channels_out,
                downsample=downsample
            )
            self.right = None
        elif level == 1:
            self.aggregator = aggregator_factory(channels_agg, channels_out)
            self.left = block_factory(
                channels_in,
                channels_out,
                downsample=downsample
            )
            self.right = block_factory(
                channels_out,
                channels_out,
                downsample=1
            )
        else:
            self.aggregator = None
            self.left = AggregationTreeNode(
                channels_in,
                channels_out,
                level - 1,
                block_factory,
                aggregator_factory,
                downsample=downsample
            )
            self.right = AggregationTreeNode(
                channels_out,
                channels_out,
                level - 1,
                block_factory,
                aggregator_factory,
                channels_agg = channels_agg + channels_out,
                downsample=1
            )

    def forward(self, x, pass_through=None):
        """
        Forward input through tree and aggregate results from child nodes.
        """
        if self.level == 0:
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
            downsample=1

    ):
        channels_agg = 2 * channels_out + channels_in
        super().__init__(
            channels_in,
            channels_out,
            tree_height,
            block_factory,
            aggregator_factory,
            channels_agg=channels_agg,
            downsample=downsample
        )

        self.downsampler = None
        if downsample > 1:
            self.downsampler = nn.MaxPool2d(kernel_size=downsample)

    def forward(self, x):
        """
        Forward input through tree and aggregate results from child nodes.
        """
        y_1 = self.left(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return self.right(y_1, [x, y_1])


class AggregationTreeFactory:
    """
    An aggregation tree implementing hierarchical aggregation of block in a stage.
    """
    def __init__(
            self,
            aggregator_factory
    ):
        self.aggregator_factory = aggregator_factory

    def __call__(
            self,
            channels_in,
            channels_out,
            n_blocks,
            block_factory,
            downsample=1
    ):
        n_levels = np.log2(n_blocks) + 1
        return AggregationTreeRoot(
            channels_in,
            channels_out,
            n_levels,
            block_factory,
            self.aggregator_factory,
            downsample=downsample
        )


