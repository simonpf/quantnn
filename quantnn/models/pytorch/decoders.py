"""
quantnn.models.pytorch.decoders
===============================

Provides generic decoder modules.
"""
from typing import Optional, Callable, Union, Optional, List, Dict

import numpy as np
import torch
from torch import nn

from quantnn.models.pytorch.base import ParamCount
from quantnn.models.pytorch.encoders import SequentialStageFactory, StageConfig
from quantnn.models.pytorch.aggregators import (
    SparseAggregatorFactory,
    LinearAggregatorFactory,
)
from quantnn.packed_tensor import forward
from quantnn.models.pytorch.upsampling import BilinearFactory

###############################################################################
# Upsampling module.
###############################################################################


def _determine_skip_connections(
    skip_connections: Union[bool, int, Dict[int, int]],
    channels: List[int],
    upsampling_factors: List[int],
    base_scale: int
) -> Dict[int, int]:
    """
    Calculates a dict of skip connections from either a bool, an in

    Args:
        skip_connections: A bool, int, or dictionary specifying whether
            or not the decoder has skip connections.
        channels: The list of channels in the stages of the decoder.
        upsampling_factors: The corresponding list of upsampling factors.
        base_scale: The base scale of the decoder.

    Return:
            A dicitonary that maps scales to the number of incoming
            channels.

    Raises:
        Value error is skip connections is not of any of the supported types.
    """
    if isinstance(skip_connections, dict):
        return skip_connections

    if isinstance(skip_connections, bool):
        skips = {}
        if skip_connections:
            scale = base_scale
            for f_u, chans in zip(upsampling_factors, channels):
                scale /= f_u
                skips[scale] = chans
        return skips
    elif isinstance(skip_connections, int):
        skips = {}
        if skip_connections:
            scale = base_scale
            for f_u, chans in zip(upsampling_factors[:skip_connections], channels[1:skip_connections]):
                scale /= f_u
                skips[scale] = chans
        return skips

    raise ValueError("Skip connections should be a bool, int or a dictionary.")


class SpatialDecoder(nn.Module, ParamCount):
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
        upsampler_factory: Callable[[int, int, int], nn.Module] = BilinearFactory(),
        upsampling_factors: List[int] = None,
        base_scale: Optional[int] = None,
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
            channels = [channels * channel_scaling**i for i in range(n_stages + 1)][
                ::-1
            ]
            if max_channels is not None:
                channels = [min(ch, max_channels) for ch in channels]

        if len(channels) != len(stages) + 1:
            raise ValueError(
                "The list of given channel numbers must exceed the number "
                "of stages in the decoder by one."
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
        if len(stages) != len(upsampling_factors):
            raise ValueError(
                "The number of upsampling factors  must equal to the "
                "number of stages."
            )
        self.upsampling_factors = upsampling_factors

        if base_scale is None:
            if isinstance(skip_connections, dict):
                base_scale = max(skip_connections.keys())
            else:
                base_scale = np.prod(upsampling_factors)
        self.base_scale = base_scale
        self.skip_connections = _determine_skip_connections(
            skip_connections, channels, upsampling_factors, base_scale
        )
        self.has_skips = len(self.skip_connections) > 0

        channels_in = channels[0]
        scale = self.base_scale
        for index, (config, channels_out) in enumerate(zip(stages, channels[1:])):

            scale /= upsampling_factors[index]

            self.upsamplers.append(
                upsampler_factory(
                    channels_in=channels_in,
                    channels_out=channels_in,
                    factor=upsampling_factors[index],
                )
            )

            channels_combined = channels_in + self.skip_connections.get(scale, 0)

            self.stages.append(
                stage_factory(
                    channels_combined,
                    channels_out,
                    config.n_blocks,
                    block_factory,
                )
            )
            channels_in = channels_out

    def forward_w_intermediate(self, x: Union[torch.Tensor, List[torch.Tensor]]):
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

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
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
            scale = self.base_scale
            y = x[scale]
            stages = self.stages

            for ind, (up, stage) in enumerate(zip(self.upsamplers, stages)):
                scale /= self.upsampling_factors[ind]
                if scale in self.skip_connections:
                    y = stage(torch.cat([x[scale], up(y)], dim=1))
                else:
                    y = stage(up(y))
        else:
            y = x
            stages = self.stages
            for up, stage in zip(self.upsamplers, stages):
                y = stage(up(y))
        return y


class SparseSpatialDecoder(nn.Module, ParamCount):
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
        base_scale: Optional[int] = None,
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
            channels = [channels * channel_scaling**i for i in range(n_stages + 1)][
                ::-1
            ]

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
        self.upsampling_factors = upsampling_factors

        # Determine scales of skip connections
        if base_scale is None:
            if isinstance(skip_connections, dict):
                base_scale = max(skip_connections.keys())
            else:
                base_scale = np.prod(upsampling_factors)
        self.base_scale = base_scale
        self.skip_connections = _determine_skip_connections(
            skip_connections, channels, upsampling_factors, base_scale
        )
        self.has_skips = len(self.skip_connections) > 0

        if multi_scale_output is not None:
            self.projections = nn.ModuleList()
        else:
            self.projections = None

        if stage_factory is None:
            stage_factory = SequentialStageFactory()
        if aggregator_factory is None:
            aggregator_factory = SparseAggregatorFactory(LinearAggregatorFactory())

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
            channels_out = multi_scale_output
            if channels_out == -1:
                channels_out = channels_in
            if channels_in != channels_out:
                self.projections.append(
                    nn.Sequential(
                        nn.Conv2d(channels_in, channels_out, kernel_size=1),
                    )
                )
            else:
                self.projections.append(nn.Identity())

        scale = base_scale
        for index, (config, channels_out) in enumerate(zip(stages, channels[1:])):
            self.upsamplers.append(
                upsampler_factory(
                    channels_in=channels_in,
                    channels_out=channels_out,
                    factor=upsampling_factors[index],
                )
            )
            scale //= upsampling_factors[index]
            if scale in self.skip_connections:
                self.aggregators[str(scale)] = aggregator_factory(
                    (channels_out, self.skip_connections[scale]), channels_out
                )

            self.stages.append(
                stage_factory(
                    channels_out,
                    channels_out,
                    config.n_blocks,
                    block_factory,
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

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
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

        scale = self.base_scale

        if isinstance(x, dict):

            y = x[scale]
            stages = self.stages
            if self.projections is not None:
                results.append(self.projections[0](y))

            for ind, (up, stage) in enumerate(zip(self.upsamplers, stages)):
                scale //= self.upsampling_factors[ind]
                y_up = forward(up, y)
                if scale in x:
                    agg = self.aggregators[str(scale)]
                    y = forward(stage, agg(y_up, x[scale]))
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


class DLADecoderStage(nn.Module, ParamCount):
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
        aggregator_factory: Callable[[int, int, int], nn.Module],
        upsampler_factory: Callable[[int, int, int], nn.Module],
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
                    channels_in=ch_in_1,
                    channels_out=ch_in_1,
                    factor=f_up,
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


class DLADecoder(nn.Sequential, ParamCount):
    """
    The decoder proposed in the deep layer-aggregation paper.

    This decoder iteratively upsamples and aggregates multi-scale
    features until the highest scale is reached.
    """

    def __init__(
        self,
        inputs: List[int],
        scales: List[int],
        aggregator_factory: Callable[[int, int, int], nn.Module],
        upsampler_factory: Callable[[int, int, int], nn.Module],
        channels: Optional[List[int]] = None,
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
                    upsampler_factory,
                )
            )
            ch_in = ch_out
        super().__init__(*blocks)

    def forward(self, x):
        """Propagate input through decoder."""
        x = [x[ind - 1] for ind in range(len(x), 0, -1)]
        return super().forward(x)[0]
