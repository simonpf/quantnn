"""
quantnn.models.pytorch.decoders
===============================

Provides generic decoder modules.
"""
from typing import Optional, Callable, Union, Optional, List

import torch
from torch import nn

from quantnn.models.pytorch.encoders import (
    SequentialStageFactory,
    StageConfig
)

###############################################################################
# Upsampling module.
###############################################################################


class Bilinear:
    """
    A factory for producing bilinear upsampling layers.
    """
    def __call__(self, factor):
        return nn.UpsamplingBilinear2d(scale_factor=factor)


###############################################################################
# Decoders.
###############################################################################


class SpatialDecoder(nn.Module):
    """
    A decoder for spatial information.

    The decoder takes a 4D input (batch x channel x height x width),
    and decoder the information into an output with usually less
    channels but reduced height and width.
    """
    def __init__(
            self,
            output_channels: int,
            stages: List[Union[int, StageConfig]],
            block_factory: Optional[Callable[[int, int], nn.Module]],
            channel_scaling: int = 2,
            max_channels: int = None,
            skip_connections: Union[bool, int] = False,
            stage_factory: Optional[Callable[[int, int], nn.Module]] = None,
            upsampler_factory: Callable[[int], nn.Module] = Bilinear(),
    ):
        super().__init__()
        n_stages = len(stages)
        self.n_stages = n_stages
        self.upsamplers = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.skip_connections = skip_connections

        input_channels = output_channels * channel_scaling ** n_stages

        if stage_factory is None:
            stage_factory = SequentialStageFactory()

        try:
            stages = [
                stage if isinstance(stage, StageConfig) else StageConfig(stage)
                for stage in stages
            ]
        except ValueError:
            raise ValueError(
                "'stages' must be a list of 'StageConfig' or 'int'  objects."
            )

        channels = input_channels
        for index, config in enumerate(stages):
            if max_channels is None:
                channels_in = channels
                channels_out = channels // channel_scaling
            else:
                channels_in = min(channels, max_channels)
                channels_out = min(channels // channel_scaling, max_channels)

            self.upsamplers.append(
                upsampler_factory(2)
            )
            channels_combined = channels_in
            if type(self.skip_connections) == bool and self.skip_connections:
                channels_combined += channels_out
            if type(self.skip_connections) == int and index < self.skip_connections:
                channels_combined += channels_out
            self.stages.append(
                stage_factory(
                    channels_combined,
                    channels_out,
                    config.n_blocks,
                    block_factory,
                    downsample=False
                )
            )
            channels = channels // channel_scaling

    def forward(
            self,
            x: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Args:
            x: The output from the encoder. This should be a single tensor
               if no skip connections are used. If skip connections are used,
               x should be list containing the outputs from each stage in the
               encoder.

        Return:
            The output tensor from the last decoder stage.
        """
        if self.skip_connections:
            if not isinstance(x, list):
                raise ValueError(
                    f"For a decoder with skip connections the input must "
                    f"be a list of tensors."
                )

        if isinstance(x, list):
            if len(x) < self.n_stages + 1:
                x = [None] * (self.n_stages + 1 - len(x)) + x
            y = x[-1]
            for x_skip, up, stage in zip(x[-2::-1], self.upsamplers, self.stages):
                if x_skip is None:
                    y = stage(up(y))
                else:
                    y = stage(torch.cat([x_skip, up(y)], dim=1))
        else:
            y = x
            for up, stage in zip(self.upsamplers, self.stages):
                y = stage(up(y))
        return y
