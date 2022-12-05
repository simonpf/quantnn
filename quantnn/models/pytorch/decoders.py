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
from quantnn.models.pytorch.aggregators import (
    SparseAggregatorFactory,
    LinearAggregatorFactory
)
from quantnn.packed_tensor import forward
from quantnn.models.pytorch.upsampling import BilinearFactory

###############################################################################
# Upsampling module.
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
            channels: Union[List[int], int],
            stages: List[Union[int, StageConfig]],
            block_factory: Optional[Callable[[int, int], nn.Module]],
            channel_scaling: int = 2,
            max_channels: int = None,
            skip_connections: Union[bool, int] = False,
            stage_factory: Optional[Callable[[int, int], nn.Module]] = None,
            upsampler_factory: Callable[[int], nn.Module] = BilinearFactory(),
            upsampling_factors: List[int] = None
    ):
        """
        Args:
            channels: A list specifying the channels before and after all
                stages in the decoder. Alternatively, channels can be an
                integer specifying the number of channels of the output from
                the last stage of the encoder. In this case, the number
                of channels of previous stages is computed by scaling
                the number of channels in each stage with 'channel_scaling'.
            block_factory: Factory functional to use to create the blocks
                in the encoders' stages.
            channel_scaling: Factor specifying the decrease in channels with
                each stage.
            max_channels: Cutoff value to limit the number of channels. Only
                used if channels is an integer.
            skip_connections: If bool, should specify whether the decoder
                should make use of skip connections. If an ``int`` it should
                specify up to which stage of the decoder skip connections will
                be provided. This can be used to implement a decoder with
                skip connects that yields a higher resolution than the input
                encoder.
            stage_factory: Factory functional to use to create the stages in
                the decoder.
            upsampler_factory: Factory functional to use to create the
                upsampling modules in the decoder.
            upsampling_factors: The upsampling factors for each decoder
                stage.
        """
        super().__init__()
        n_stages = len(stages)
        self.n_stages = n_stages
        self.upsamplers = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.skip_connections = skip_connections

        if isinstance(channels, int):
           channels = [
               channels * channel_scaling ** i for i in range(n_stages + 1)
           ][::-1]
           if max_channels is not None:
               channels = [min(ch, max_channels) for ch in channels]

        if not len(channels) == len(stages) + 1:
            raise ValueError(
                "The list of given channel numbers must match the number "
                "of stages plus 1."
            )

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

        if upsampling_factors is None:
            upsampling_factors = [2] * n_stages
        if not len(stages) == len(upsampling_factors):
            raise ValueError(
                "The list of upsampling factors  numbers must match the number "
                "of stages."
            )

        channels_in = channels[0]
        for index, (config, channels_out) in enumerate(zip(stages, channels[1:])):

            self.upsamplers.append(
                upsampler_factory(upsampling_factors[index])
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
            channels_in = channels_out

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


class SparseSpatialDecoder(nn.Module):
    """
    A decoder for spatial information which supports missing skip
    connections.

    The decoder uses explicit aggregation block that combine the inputs
    from the encoder with the decoders own upsampling data stream. By
    using when a SparseAggregator, the decoder can handle cases
    where certain input samples are missing from the skip connections.
    """
    def __init__(
            self,
            channels: int,
            stages: List[Union[int, StageConfig]],
            block_factory: Optional[Callable[[int, int], nn.Module]],
            channel_scaling: int = 2,
            max_channels: int = None,
            skip_connections: int = -1,
            multi_scale_output: int = None,
            stage_factory: Optional[Callable[[int, int], nn.Module]] = None,
            upsampler_factory: Callable[[int], nn.Module] = BilinearFactory(),
            aggregator_factory: Callable[[int, int], nn.Module] = None,
            upsampling_factors: List[int] = None
    ):
        """
        Args:
            channels: A list specifying the channels before and after all
                stages in the decoder. Alternatively, channels can be an
                integer specifying the number of channels of the output from
                the last stage of the encoder. In this case, the number
                of channels of previous stages is computed by scaling
                the number of channels in each stage with 'channel_scaling'.
            block_factory: Factory functional to use to create the blocks
                in the encoders' stages.
            channel_scaling: Factor specifying the decrease in channels with
                each stage.
            max_channels: Cutoff value to limit the number of channels. Only
                used if channels is an integer.
            skip_connections: If bool, should specify whether the decoder
                should make use of skip connections. If an ``int`` it should
                specify up to which stage of the decoder skip connections will
                be provided. This can be used to implement a decoder with
                skip connects that yields a higher resolution than the input
                encoder.
            stage_factory: Factory functional to use to create the stages in
                the decoder.
            upsampler_factory: Factory functional to use to create the
                upsampling modules in the decoder.
            aggregator_factory: Factory functional to create aggregation
                blocks.
            upsampling_factors: The upsampling factors for each decoder
                stage.
        """
        super().__init__()
        n_stages = len(stages)
        self.n_stages = n_stages
        self.upsamplers = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.aggregators = nn.ModuleList()
        if skip_connections < 0:
            skip_connections = n_stages
        self.skip_connections = skip_connections

        if isinstance(channels, int):
           channels = [
               channels * channel_scaling ** i for i in range(n_stages + 1)
           ][::-1]
           if max_channels is not None:
               channels = [min(ch, max_channels) for ch in channels]

        if not len(channels) == len(stages) + 1:
            raise ValueError(
                "The list of given channel numbers must match the number "
                "of stages plus 1."
            )

        if upsampling_factors is None:
            upsampling_factors = [2] * n_stages
        if not len(stages) == len(upsampling_factors):
            raise ValueError(
                "The list of upsampling factors  numbers must match the number "
                "of stages."
            )

        if multi_scale_output is not None:
            self.projections = nn.ModuleList()
        else:
            self.projections = None

        if stage_factory is None:
            stage_factory = SequentialStageFactory()
        if aggregator_factory is None:
            aggregator_factory = SparseAggregatorFactory(
                LinearAggregatorFactory()
            )

        try:
            stages = [
                stage if isinstance(stage, StageConfig) else StageConfig(stage)
                for stage in stages
            ]
        except ValueError:
            raise ValueError(
                "'stages' must be a list of 'StageConfig' or 'int'  objects."
            )

        channels_in = channels[0]
        for index, (config, channels_out) in enumerate(zip(stages, channels[1:])):
            self.upsamplers.append(
                upsampler_factory(
                    upsampling_factors[index],
                    channels_in,
                    channels_out
                )
            )
            if index < self.skip_connections:
                self.aggregators.append(
                    aggregator_factory(
                        channels_out, 2, channels_out
                    )
                )
            else:
                self.aggregators.append(None)

            self.stages.append(
                stage_factory(
                    channels_out,
                    channels_out,
                    config.n_blocks,
                    block_factory,
                    downsample=False
                )
            )
            if self.projections is not None:
                if channels_out != multi_scale_output:
                    self.projections.append(
                        nn.Sequential(
                            nn.Conv2d(channels_out, multi_scale_output, kernel_size=1),
                        )
                    )
                else:
                    self.projections.append(
                        nn.Sequential(
                            nn.Identity(),
                        )
                    )
            channels_in = channels_out

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
        if not isinstance(x, list):
            raise ValueError(
                f"For a decoder with skip connections the input must "
                f"be a list of tensors."
            )

        # Pad input with None
        if len(x) < self.n_stages + 1:
            x = [None] * (self.n_stages + 1 - len(x)) + x

        y = x[-1]
        results = []
        for ind, (x_skip, up, agg, stage) in enumerate(zip(
                x[-2::-1],
                self.upsamplers,
                self.aggregators,
                self.stages
        )):
            y_up = forward(up, y)
            if x_skip is None:
                y = forward(stage, y_up)
            else:
                y_agg = agg(y_up, x_skip)
                y = forward(stage, y_agg)
            if self.projections is not None:
                results.append(self.projections[ind](y))
        if self.projections is not None:
            return results
        return y


class DLADecoderStage(nn.Module):
    """
    A single stage of a DLA decoder.

    Each stage of the DLA decoder acts in parallel on a list of
    multi-scale inputs. Each input is upsampled to the subsequent
    lower scale and merged with the input at that scale.
    """
    def __init__(
            self,
            channels: List[int],
            scales: List[int],
            aggregator_factory: Callable[[int, int,int], nn.Module],
            upsampler_factory: Callable[[int, int, int], nn.Module]
    ):
        """
        Args:
            channels: List the input channels at each scale.
            scales: The scales of each input with the largest scale
                as the first element.
            aggregator_factory: A factory functional to use to create
                aggregation blocks.
            upsampler_factory: A factory functional to use to create
                upsampling blocks.
        """
        super().__init__()
        n_scales = len(scales)
        upsamplers = []
        aggregators = []
        for scale_index in range(n_scales - 1):
            f_up = scales[scale_index] / scales[scale_index + 1]
            ch_in_1 = channels[scale_index]
            ch_in_2 = channels[scale_index + 1]

            upsamplers.append(
                upsampler_factory(
                    f_up,
                    channels_in=ch_in_1,
                    channels_out=ch_in_2
                )
            )

            aggregators.append(aggregator_factory(ch_in_2, 2, ch_in_2))

        self.upsamplers = nn.ModuleList(upsamplers)
        self.aggregators = nn.ModuleList(aggregators)

    def forward(self, x):
        """
        Propagate multi-scale inputs through decoder.

        Args:
            x: A list of multi-scale input containing the coarsest scale
                as the first element.

        Return:
            A list containing the outputs from the decoder stage.
        """
        results = []
        for scale_index in range(len(x) - 1):
            up = self.upsamplers[scale_index]
            x_1 = forward(up, x[scale_index])
            agg = self.aggregators[scale_index]
            x_2 = x[scale_index + 1]
            results.append(agg(x_1, x_2))
        return results


class DLADecoder(nn.Sequential):
    """
    The decoder proposed in the deep layer-aggregation paper.

    This decoder iteratively upsamples and aggregates multi-scale
    features until the highest scale is reached.
    """
    def __init__(
            self,
            channels: List[int],
            scales: List[int],
            aggregator_factory: Callable[[int, int,int], nn.Module],
            upsampler_factory: Callable[[int, int, int], nn.Module]
    ):
        blocks = []
        for scale_index in range(len(scales) - 1):
            blocks.append(
                DLADecoderStage(
                    channels[scale_index:],
                    scales[scale_index:],
                    aggregator_factory,
                    upsampler_factory
                )
            )
        super().__init__(*blocks)

    def forward(self, x):
        return super().forward(x)[0]





