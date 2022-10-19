"""
quantnn.models.pytorch.encoders
===============================

Generic encoder modules.
"""
from dataclasses import dataclass
from math import log
from typing import Optional, Callable, Union, Optional, List, Dict

import torch
from torch import nn


@dataclass
class StageConfig:
    """
    Configuration of a single stage in an encoder.
    """
    n_blocks: int
    block_args: Optional[List] = lambda: []
    block_kwargs: Optional[List] = lambda: []


class SequentialStageFactory(nn.Sequential):
    """
    A stage consisting of a simple sequence of blocks.
    """
    def __call__(
            self,
            channels_in,
            channels_out,
            n_blocks,
            block_factory,
            downsample=False
    ):
        """
        Args:
            channels_in: The number of channels in the input to the
                first block.
            channels_out: The number of channels in the input to
                all other blocks an the output from the last block.
            n_blocks: The number of blocks in the stage.
            block_factory: The factory to use to create the blocks
                in the stage.
            downsample: Whether to include a downsampling layer
                in the stage.
        """
        blocks = []
        for _ in range(n_blocks):
            blocks.append(
                block_factory(
                    channels_in,
                    channels_out,
                    downsample=downsample
                )
            )
            channels_in = channels_out
            downsample = False
        return nn.Sequential(*blocks)


class SpatialEncoder(nn.Module):
    """
    An encoder for spatial information.

    The encoder takes a 4D input (batch x channel x height x width),
    and encodes the information into an output with a higher number
    of channels but reduced height and width.
    """
    def __init__(
            self,
            input_channels: int,
            stages: List[Union[int, StageConfig]],
            block_factory: Optional[Callable[[int, int], nn.Module]],
            channel_scaling: int = 2,
            max_channels: int = None,
            stage_factory: Optional[Callable[[int, int], nn.Module]] = None,
            downsampler_factory: Callable[[int, int], nn.Module] = None,
    ):
        """
        Args:
            input_channels: The number of input channels to the encoder.
            stages: A list containing the stage specifications for each
                stage in the encoder.
            block_factory: Factory to create the blocks in each stage.
            channel_scaling: Scaling factor specifying the increase of the
                number of channels after every downsampling layer.
            max_channels: Cutoff value to limit the number of channels.
            stage_factory: Optional stage factory to create the encoder
                stages. Defaults to ``SequentialStageFactory``.
            downsampler_factory: Optional factory to create downsampling
                layers. If not provided, the block factory must provide
                downsampling functionality.
        """
        super().__init__()
        self.channel_scaling = channel_scaling
        self.downsamplers = nn.ModuleList()
        self.stages = nn.ModuleList()
        channels = input_channels

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


        for stage in stages:
            if max_channels is None:
                channels_out = channels * channel_scaling
            else:
                channels_out = min(channels * channel_scaling, max_channels)

            # Downsampling layer is included in stage.
            if downsampler_factory is None:
                self.downsamplers.append(None)
                self.stages.append(
                    stage_factory(
                        channels,
                        channels_out,
                        stage.n_blocks,
                        block_factory,
                        downsample=True
                    )
                )
            # Explicit downsampling layer.
            else:
                self.downsamplers.append(
                    downsampler_factory(2)
                )
                self.stages.append(
                    stage_factory(
                        channels_out,
                        channels_out,
                        stage.n_blocks,
                        block_factory,
                        downsample=False
                    )
                )
            channels = channels_out

    def forward_with_skips(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: A ``torch.Tensor`` to feed into the encoder.

        Return:
            A list containing the outputs of each encoder stage with the
            last element in the list corresponding to the output of the
            last encoder stage.
        """
        y = x
        skips = [y]
        for down, stage in zip(self.downsamplers, self.stages):
            if down is not None:
                y = down(y)
            y = stage(y)
            skips.append(y)
        return skips

    def forward(
            self,
            x: torch.Tensor,
            return_skips: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: A ``torch.Tensor`` to feed into the encoder.
            return_skips: Whether or not the feature maps from all
                stages of the encoder should be returned.

        Return:
            If ``return_skips`` is ``False`` only the feature maps output
            by the last encoder stage are returned. Otherwise a list containing
            the feature maps from all stages are returned with the last element
            in the list corresponding to the output of the last encoder stage.
        """
        if return_skips:
            return self.forward_with_skips(x)
        y = x
        for down, stage in zip(self.downsamplers, self.stages):
            if down is not None:
                y = down(y)
            y = stage(y)
        return y


class MultiInputSpatialEncoder(SpatialEncoder):
    """
    An encoder for spatial information with multiple inputs at
    different stages.

    The encoder takes a 4D input (batch x channel x height x width),
    and encodes the information into an output with a higher number
    of channels but reduced height and width.
    """
    def __init__(
            self,
            input_channels: Dict[int, int],
            base_channels: int,
            stages: List[Union[int, StageConfig]],
            block_factory: Optional[Callable[[int, int], nn.Module]],
            aggregator_factory: Optional[Callable[[int], nn.Module]],
            channel_scaling: int = 2,
            max_channels: int = None,
            stage_factory: Optional[Callable[[int, int], nn.Module]] = None,
            downsampler_factory: Callable[[int, int], nn.Module] = None,
    ):
        n_stages = len(stages)
        if max_channels is None:
            max_channels = base_channels * channel_scaling ** n_stages

        keys = list(input_channels.keys())
        first_stage = min(keys)
        true_input_channels = min(
            base_channels * channel_scaling ** first_stage,
            max_channels
        )
        super().__init__(
            true_input_channels,
            stages[first_stage:],
            block_factory,
            channel_scaling=channel_scaling,
            max_channels=max_channels,
            stage_factory=stage_factory,
            downsampler_factory=downsampler_factory
        )
        self.stems = nn.ModuleDict()
        self.aggregators = nn.ModuleDict()
        self.input_channels = input_channels
        self.first_stage = first_stage

        for ind, (stage_ind, channels_in) in enumerate(input_channels.items()):
            channels_out = min(
                base_channels * channel_scaling ** stage_ind,
                max_channels)
            self.stems[str(stage_ind)] = block_factory(channels_in, channels_out)
            if ind > 0:
                self.aggregators[str(stage_ind)] = aggregator_factory(
                    channels_out,
                    2,
                    channels_out,
                )


    def forward_with_skips(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: A ``torch.Tensor`` to feed into the encoder.

        Return:
            A list containing the outputs of each encoder stage with the
            last element in the list corresponding to the output of the
            last encoder stage.
        """
        stage_ind = self.first_stage
        skips = []
        y = None
        input_index = 0

        for down, stage in zip(self.downsamplers, self.stages):
            # Integrate input into stream.
            stage_s = str(stage_ind)
            if stage_ind in self.input_channels:
                if y is None:
                    y = self.stems[stage_s](x[input_index])
                    skips.append(y)
                else:
                    agg = self.aggregators[stage_s]
                    y = agg(y, self.stems[stage_s](x[input_index]))
                input_index += 1

            if down is not None:
                y = down(y)
            y = stage(y)
            skips.append(y)
            stage_ind += 1
        return skips

    def forward(
            self,
            x: torch.Tensor,
            return_skips: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: A ``torch.Tensor`` to feed into the encoder.
            return_skips: Whether or not the feature maps from all
                stages of the encoder should be returned.

        Return:
            If ``return_skips`` is ``False`` only the feature maps output
            by the last encoder stage are returned. Otherwise a list containing
            the feature maps from all stages are returned with the last element
            in the list corresponding to the output of the last encoder stage.
        """
        if not len(x) == len(self.input_channels):
            raise ValueError(
                "Multi-input encoder expects a list of tensors with the same "
                " number of elements as the 'input_channels' dict."
            )

        if return_skips:
            return self.forward_with_skips(x)

        stage_ind = self.first_stage
        y = None
        input_index = 0

        for down, stage in zip(self.downsamplers, self.stages):
            # Integrate input into stream.
            stage_s = str(stage_ind)
            if stage_ind in self.input_channels:
                if y is None:
                    y = self.stems[stage_s](x[input_index])
                else:
                    agg = self.aggregators[stage_s]
                    y = agg(y, self.stems[stage_s](x[input_index]))
                input_index += 1

            if down is not None:
                y = down(y)
            y = stage(y)
            stage_ind = stage_ind + 1
        return y
