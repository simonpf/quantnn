"""
quantnn.models.pytorch.decoders
===============================

Provides generic decoder modules.
"""
from typing import Optional, Callable, Union, Optional, List, Dict

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

def _determine_skip_connections(
        skip_connections: Union[bool, int, Dict[int, int]],
        channels: List[int],
        base_scale
) -> Dict[int, int]:
    """
    Calculates a dict of skip connections from either a bool, an in

    Args:
        skip_connections: A bool, int, or dictionary specifying whether
            or not the decoder has skip connections.
        channels: The list of channels in the stages of the decoder.

    Return:
            A dicitonary that map stage indices to the number of incoming
            channels.

    Raises:
        Value error is skip connections is not of any of the supported types.
    """
    if isinstance(skip_connections, dict):
        return skip_connections

    if isinstance(skip_connections, bool):
        if skip_connections:
            return {
                base_scale - ind - 1: ch for ind, ch in enumerate(channels[1:])
            }
        else:
            return {}
    elif isinstance(skip_connections, int):
        skip_chans = channels[1: skip_connections + 1]
        return {
            base_scale - ind - 1: ch for ind, ch in enumerate(skip_chans)
        }


    raise ValueError(
        "Skip connections should be a bool, int or a dictionary."
    )


class SpatialDecoder(nn.Module):
    """
    A decoder for spatial information.

    The decoder takes a 4D input (batch x channel x height x width),
    and decodes channel information input spatial information.

    The decoder consists of multiple stages each preceded by an
    upsampling layer. Features from skip connections are merged
    after the upsamling before the convolutional block of each stage
    are applied.
    """
    def __init__(
            self,
            channels: Union[List[int], int],
            stages: List[Union[int, StageConfig]],
            block_factory: Optional[Callable[[int, int], nn.Module]],
            channel_scaling: int = 2,
            max_channels: int = None,
            skip_connections: Union[bool, int, Dict[int, int]] = False,
            stage_factory: Optional[Callable[[int, int], nn.Module]] = None,
            upsampler_factory: Callable[[int], nn.Module] = BilinearFactory(),
            upsampling_factors: List[int] = None,
            base_scale : Optional[int] = None
    ):
        """
        Args:
            channels: A list specifying the channels before the first stage
                and within all consecutive stages. Alternatively, this can
                be just an integer specifying the number of channels of
                the input to the decoder. The number of channels within
                each stage will then be computed by multiplying the input
                channels with increasing powers of the 'channel_scaling'
                parameter.
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
                skip connections that yields a higher resolution than the input
                encoder. If a 'dict' it should map stage indices to the
                number of channels in the skip connections.
            stage_factory: Factory functional to use to create the stages in
                the decoder.
            upsampler_factory: Factory functional to use to create the
                upsampling modules in the decoder.
            upsampling_factors: The upsampling factors for each decoder
                stage.
            base_scale: The index of the deepest stage in the encoder. This
                is required for encoder-decoder combinations in which the
                number of stages in en- and decoder are different. If None,
                the number of stages in en- and decoder are assumed to be
                the same.
        """
        super().__init__()
        n_stages = len(stages)
        self.n_stages = n_stages
        self.upsamplers = nn.ModuleList()
        self.stages = nn.ModuleList()


        if isinstance(channels, int):
           channels = [
               channels * channel_scaling ** i for i in range(n_stages + 1)
           ][::-1]
           if max_channels is not None:
               channels = [min(ch, max_channels) for ch in channels]

        if len(channels) != len(stages) + 1:
            raise ValueError(
                "The list of given channel numbers must exceed the number "
                "of stages in the decoder by one."
            )

        if base_scale is None:
            if isinstance(skip_connections, dict):
                base_scale = max(skip_connections.keys())
            else:
                base_scale = len(stages)
        self.base_scale = base_scale
        self.skip_connections = _determine_skip_connections(
            skip_connections,
            channels,
            base_scale
        )
        self.has_skips = len(self.skip_connections) > 0

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
        if len(stages) != len(upsampling_factors):
            raise ValueError(
                "The number of upsampling factors  must equal to the "
                "number of stages."
            )

        channels_in = channels[0]
        for index, (config, channels_out) in enumerate(zip(stages, channels[1:])):

            stage_ind = base_scale - index - 1

            self.upsamplers.append(
                upsampler_factory(upsampling_factors[index])
            )

            channels_combined = (
                channels_in + self.skip_connections.get(stage_ind, 0)
            )

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

    def forward_w_intermediate(
            self,
            x: Union[torch.Tensor, List[torch.Tensor]]
    ):
        """
        Same as 'forward' but also returns the intermediate activation
        from after each stage.

        Args:
            x: The output from the encoder. This should be a single tensor
               if no skip connections are used. If skip connections are used,
               x should be list containing the outputs from each stage in the
               encoder.

        Return:
            A list of tensors containing the activations after every stage
            in the decoder.
        """
        if self.has_skips:
            if not isinstance(x, dict):
                raise ValueError(
                    f"For a decoder with skip connections the input must "
                    f"be a dictionary mapping stage indices to inputs."
                )
        else:
            if isinstance(x, dict):
                x = x[self.n_stages]

        activations = []

        if isinstance(x, dict):
            y = x[self.base_scale]
            stages = self.stages

            for ind, (up, stage) in enumerate(zip(self.upsamplers, stages)):
                stage_ind = self.base_scale - ind - 1
                if stage_ind in self.skip_connections:
                    y = stage(torch.cat([x[stage_ind], up(y)], dim=1))
                else:
                    y = stage(up(y))
                activations.append(y)
        else:
            y = x
            stages = self.stages
            for up, stage in zip(self.upsamplers, stages):
                y = stage(up(y))
                activations.append(y)
        return activations

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
        if self.has_skips:
            if not isinstance(x, dict):
                raise ValueError(
                    f"For a decoder with skip connections the input must "
                    f"be a dictionary mapping stage indices to inputs."
                )
        else:
            if isinstance(x, dict):
                x = x[self.base_scale]


        if isinstance(x, dict):
            y = x[self.base_scale]
            stages = self.stages

            for ind, (up, stage) in enumerate(zip(self.upsamplers, stages)):
                stage_ind = self.base_scale - ind - 1
                if stage_ind in self.skip_connections:
                    y = stage(torch.cat([x[stage_ind], up(y)], dim=1))
                else:
                    y = stage(up(y))
        else:
            y = x
            stages = self.stages
            for up, stage in zip(self.upsamplers, stages):
                y = stage(up(y))
        return y


class SparseSpatialDecoder(nn.Module):
    """
    A decoder for spatial information which supports missing skip
    connections.

    The decoder uses explicit aggregation block that combine the inputs
    from the encoder with the decoders own upsampling data stream. By
    using a SparseAggregator, the decoder can handle cases
    where certain input samples are missing from the skip connections.
    """
    def __init__(
            self,
            channels: int,
            stages: List[Union[int, StageConfig]],
            block_factory: Optional[Callable[[int, int], nn.Module]],
            channel_scaling: int = 2,
            max_channels: int = None,
            skip_connections: int = False,
            multi_scale_output: int = None,
            stage_factory: Optional[Callable[[int, int], nn.Module]] = None,
            upsampler_factory: Callable[[int], nn.Module] = BilinearFactory(),
            aggregator_factory: Optional[Callable[[int, int], nn.Module]] = None,
            upsampling_factors: Optional[List[int]] = None,
            base_scale: Optional[int] = None
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
                skip connections that yields a higher resolution than the input
                encoder. If a 'dict' it should map stage indices to the
                number of channels in the skip connections.
            stage_factory: Factory functional to use to create the stages in
                the decoder.
            upsampler_factory: Factory functional to use to create the
                upsampling modules in the decoder.
            aggregator_factory: Factory functional to create aggregation
                blocks.
            upsampling_factors: The upsampling factors for each decoder
                stage.
            base_scale: The index of the deepest stage in the encoder. This
                is required for encoder-decoder combinations in which the
                number of stages in en- and decoder are different. If None,
                the number of stages in en- and decoder are assumed to be
                the same.
        """
        super().__init__()
        n_stages = len(stages)
        self.n_stages = n_stages
        self.upsamplers = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.aggregators = nn.ModuleDict()

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

        if base_scale is None:
            if isinstance(skip_connections, dict):
                base_scale = max(skip_connections.keys())
            else:
                base_scale = len(stages)
        self.base_scale = base_scale
        self.skip_connections = _determine_skip_connections(
            skip_connections,
            channels,
            base_scale
        )
        self.has_skips = len(self.skip_connections) > 0

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

        if self.projections is not None:
            if channels_in != multi_scale_output:
                self.projections.append(
                    nn.Sequential(
                        nn.Conv2d(channels_in, multi_scale_output, kernel_size=1),
                    )
                )
            else:
                self.projections.append(
                    nn.Identity()
                )

        for index, (config, channels_out) in enumerate(zip(stages, channels[1:])):
            self.upsamplers.append(
                upsampler_factory(
                    upsampling_factors[index],
                    channels_in,
                    channels_out
                )
            )
            stage_ind = base_scale - index - 1
            if stage_ind in self.skip_connections:
                self.aggregators[str(stage_ind)] = aggregator_factory(
                    (channels_out, self.skip_connections[stage_ind]), channels_out
                )

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
            x: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x: The output from the encoder. This should be a be list
               containing the outputs from each stage in the encoder.

        Return:
            The output tensor from the last decoder stage.
        """
        if self.has_skips:
            if not isinstance(x, dict):
                raise ValueError(
                    f"For a decoder with skip connections the input must "
                    f"be a dictionary mapping stage indices to inputs."
                )
        else:
            if isinstance(x, dict):
                x = x[self.base_scale]

        results = []

        if isinstance(x, dict):

            y = x[self.base_scale]
            stages = self.stages
            if self.projections is not None:
                results.append(self.projections[0](y))

            for ind, (up, stage) in enumerate(zip(self.upsamplers, stages)):
                stage_ind = self.base_scale - ind - 1
                y_up = forward(up, y)
                if stage_ind in x:
                    agg = self.aggregators[str(stage_ind)]
                    y = forward(stage, agg(y_up, x[stage_ind]))
                else:
                    y = forward(stage, y_up)
                if self.projections is not None:
                    results.append(self.projections[ind + 1](y))
        else:
            y = x
            stages = self.stages
            for up, stage in zip(self.upsamplers, stages):
                y = stage(up(y))

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
            inputs: List[int],
            channels: List[int],
            scales: List[int],
            aggregator_factory: Callable[[int, int,int], nn.Module],
            upsampler_factory: Callable[[int, int, int], nn.Module]
    ):
        """
        Args:
            inputs: Dictionary mapping sc
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
            ch_in_1 = inputs[scale_index]
            ch_in_2 = channels[scale_index + 1]

            upsamplers.append(
                upsampler_factory(
                    f_up,
                    channels_in=ch_in_1,
                    channels_out=ch_in_1
                )
            )


            aggregators.append(aggregator_factory((ch_in_1, ch_in_2), ch_in_2))

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
            inputs: List[int],
            scales: List[int],
            aggregator_factory: Callable[[int, int,int], nn.Module],
            upsampler_factory: Callable[[int, int, int], nn.Module],
            channels: Optional[List[int]] = None
    ):
        """
        Create a DLA decoder.

        Args:
            inputs: List specifying the number of input channels to the
                decoder.
            scales: The scales of the inputs.
            aggregator_factory: The factory to use to create the aggregator
                modules that consecutively fuses features from larger scales
                with thos from the next-lower one.
            upsampler_factory: The factory to use to create the upsample
                blocks.
        """
        blocks = []
        if channels is None:
            channels = inputs
        ch_in = inputs
        ch_out = channels
        for scale_index in range(len(scales) - 1):
            blocks.append(
                DLADecoderStage(
                    ch_in[scale_index:],
                    ch_out[scale_index:],
                    scales[scale_index:],
                    aggregator_factory,
                    upsampler_factory
                )
            )
            ch_in = ch_out
        super().__init__(*blocks)

    def forward(self, x):
        """Propagate input through decoder."""
        x = [x[ind - 1] for ind in range(len(x), 0, -1)]
        return super().forward(x)[0]
