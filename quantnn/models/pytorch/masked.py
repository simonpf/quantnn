"""
quantnn.models.pytorch.masked
=============================

Provides torch modules that support masked tensors.
"""
import math
from typing import Optional, List, Union

import torch
from torch import nn

from quantnn.masked_tensor import MaskedTensor
from quantnn.models.pytorch import normalization


class ScaleContributions(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        mask = x.__mask__
        ctx.save_for_backward(mask)
        return torch.where(mask, 0, x)

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return torch.where(mask, 0, grad_output)


@torch.jit.script
def masked_conv_2d(
        x: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor,
        ctr_weights: torch.Tensor,
        bias: torch.Tensor,
        padding: List[int],
        dilation: List[int],
        stride: List[int],
        groups: int = 1
):

    x_m = torch.where(mask, 0.0, x)
    y = nn.functional.conv2d(
        x_m,
        weight,
        None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups
    )
    masked_frac = nn.functional.conv2d(
        mask.to(dtype=x.dtype),
        ctr_weights,
        None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=1
    )

    mask_new = masked_frac >= 1.0
    y = y / torch.where(mask_new, 1.0, (1.0 - masked_frac))
    if bias is not None:
        y = y + bias[None, :, None, None]

    return y, mask_new


class Conv2d(nn.Conv2d):
    """
    Masked version of 2D convolution.

    The results of the masked convolution is calculated using only non-masked
    pixels and results are rescaled to account for missing inputs.

    Defaults to standard convolution for non-masked tensors.
    """
    def __init__(
            self,
            channels_in: int,
            channels_out: int,
            kernel_size: int,
            **kwargs
    ):
        self._bias = None
        if "bias" in kwargs:
            self.has_bias = kwargs["bias"]
        else:
            self.has_bias = True
        kwargs["bias"] = False

        super().__init__(
            channels_in,
            channels_out,
            kernel_size,
            **kwargs,
        )

        if kernel_size in [1, (1, 1)]:
            self.is_spatial = False
            kwargs["groups"] = 1
            self.counter = nn.Conv2d(
                channels_in,
                channels_out,
                1,
                **kwargs
            )
            for param in self.counter.parameters():
                param.requires_grad = False
            self.counter.weight.fill_(1.0 / channels_in)
        else:
            self.is_spatial = True
            kwargs["padding_mode"] = "reflect"
            kwargs["groups"] = 1
            self.counter = nn.Conv2d(
                1,
                1,
                kernel_size,
                **kwargs
            )
            for param in self.counter.parameters():
                param.requires_grad = False
            self.counter.weight.fill_(1.0 / self.counter.weight.data.numel())

        if self.has_bias:
            self._bias = nn.Parameter(torch.zeros(channels_out))
        self.reset_parameters()


    def reset_parameters(self) -> None:
        """
        Override reset parameters to initialize _bias instead of bias.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self._bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self._bias, -bound, bound)


    def forward(self, x):
        """
        Propagate masked tensor through layer.
        """
        if not isinstance(x, MaskedTensor):
            if self._bias is None:
                return super().forward(x)
            else:
                return super().forward(x) + self._bias[..., None, None]

        if self.is_spatial:
            mask = x.mask.any(1, keepdims=True)
        else:
            mask = x.mask

        y, new_mask = masked_conv_2d(
            x.strip(),
            mask,
            self.weight,
            self.counter.weight,
            self._bias,
            padding = self.padding,
            dilation = self.dilation,
            groups = self.groups,
            stride = self.stride
        )

        if self.is_spatial:
            new_mask = torch.broadcast_to(new_mask, y.shape)

        return MaskedTensor(y, mask=new_mask, compressed=x.compressed)



class BatchNorm2d(nn.BatchNorm1d):
    """
    Masked version of 2D batch norm.

    The 2D batch norm will mask all input positions in which more than
    one input channel is masked.
    """
    def forward(self, x):
        x_p = torch.permute(x, (0, 2, 3, 1))
        if not isinstance(x, MaskedTensor):
            y = super().forward(x_p)
            return torch.permute(y, (0, 3, 1, 2))

        mask = x_p.mask
        x_p = x_p.strip()
        y = torch.zeros_like(x_p)
        valid = ~mask.any(-1)
        y[valid] = super().forward(x_p[valid])
        y = torch.permute(y, (0, 3, 1, 2))
        mask = torch.broadcast_to(x.mask.any(1, keepdims=True), y.shape)
        return MaskedTensor(y, mask=mask, compressed=x.compressed)


class BatchNorm1d(nn.BatchNorm1d):
    """
    Masked implementation of 1D batch norm.
    """
    def forward(self, x):

        self._check_input_dim(x)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked.add_(1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        is_masked = isinstance(x, MaskedTensor)
        # Special handling is only required during training.
        if bn_training and is_masked:
            mask = x.mask
            compressed = x.compressed

            x = torch.Tensor(x)

            n_valid = (~mask).sum((0,))
            x_m = torch.where(mask, 0, x)
            mean = x_m.sum((0,)) / n_valid
            var = (x_m ** 2).sum((0,)) / n_valid - mean ** 2
            y = (x - mean) / (var + self.eps).sqrt()
            y = torch.where(mask, 0.0, y)

            if self.weight is not None:
                y = self.weight * y + self.bias

            if self.training or self.track_running_stats:
                c_1 = 1.0 - exponential_average_factor
                c_2 = exponential_average_factor
                self.running_mean = torch.where(
                    n_valid > 0,
                    c_1 * self.running_mean + c_2 * mean,
                    self.running_mean
                )
                self.running_var = torch.where(
                    n_valid > 0,
                    c_1 * self.running_var + c_2 * var,
                    self.running_var
                )
                y = MaskedTensor(y, mask=mask, compressed=compressed)
        else:
            y = nn.functional.batch_norm(
                x,
                self.running_mean
                if not self.training or self.track_running_stats
                else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,
                )

        return y


class LayerNormFirst(normalization.LayerNormFirst):
    """
    Layer norm that normalizes the first dimension
    """

    def __init__(
            self,
            n_channels,
            eps=1e-6
    ):
        """
        Args:
            n_channels: The number of channels in the input.
            eps: Epsilon added to variance to avoid numerical issues.
        """
        super().__init__(n_channels, eps=eps)
        self.scaling = nn.Parameter(torch.ones(n_channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(n_channels), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        """
        Apply normalization to x.
        """
        if not isinstance(x, MaskedTensor):
            return super().forward(x)

        mask = x.mask
        compressed = x.compressed
        x = x.strip()

        n_valid = (1 - mask.to(x.dtype)).sum(1, keepdims=True)
        x_m = torch.where(mask, 0.0, x)
        x_s = x_m.sum(1, keepdims=True)
        mu = x_s / n_valid
        x2_s = (x_m ** 2).sum(1, keepdims=True)
        var = x2_s / n_valid - mu ** 2

        x_n = torch.where(mask, 0.0, (x_m - mu) / torch.sqrt(var + self.eps))
        x = self.scaling[..., None, None] * x_n + self.bias[..., None, None]

        return MaskedTensor(x, mask=mask, compressed=compressed)


class Linear(nn.Linear):
    """
    Masked linear layer.
    """
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None
    ):
        self.sep_bias = None
        super().__init__(
            in_features,
            out_features,
            bias=False,
            device=device,
            dtype=dtype
        )
        self.counter = nn.Linear(
            in_features,
            out_features,
            bias=False,
            device=device,
            dtype=dtype
        )
        for param in self.counter.parameters():
            param.requires_grad = False
        self.counter.weight.fill_(1.0 / in_features)

        if bias:
            self.sep_bias = nn.Parameter(
                torch.zeros(out_features)
            )
        else:
            self.sep_bias = None
        self.reset_parameters()


    def forward(
            self,
            x: Union[MaskedTensor, torch.Tensor]
    ) -> Union[MaskedTensor, torch.Tensor]:
        """
        Forward input through layer.
        """
        if not isinstance(x, MaskedTensor):
            if self.sep_bias is None:
                return super().forward(x)
            else:
                return super().forward(x) + self.sep_bias

        mask = x.mask
        x_m = torch.where(mask, 0.0, x.strip())
        y = super().forward(x_m)

        masked_frac = self.counter(mask.to(dtype=x.dtype))
        mask_new = masked_frac >= 1.0

        y = y / torch.where(mask_new, 1.0, (1.0 - masked_frac))
        if self.sep_bias is not None:
            y = y + self.sep_bias

        return MaskedTensor(y, mask=mask_new, compressed=x.compressed)

    def reset_parameters(self) -> None:
        """
        Override initialization to initialize separate bias layer.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.sep_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.sep_bias, -bound, bound)


class AvgPool2d(nn.AvgPool2d):
    """
    Masked implementation of 2D batch norm.
    """
    def forward(self, x):
        if not isinstance(x, MaskedTensor):
            return super().forward(x)

        mask = x.mask
        x_m = torch.where(mask, 0.0, x.strip())
        y = super().forward(x_m)
        cts = super().forward(mask.to(dtype=x_m.dtype)).detach()
        new_mask = ~(cts < 0.5)
        return MaskedTensor(y, mask=new_mask, compressed=x.compressed)
        #return MaskedTensor(y / cts, mask=new_mask, compressed=x.compressed)


class MaxPool2d(nn.MaxPool2d):
    def forward(self, x):
        if not isinstance(x, MaskedTensor):
            return super().forward(x)

        mask = x.mask
        x_m = torch.where(mask, -torch.inf, x.strip())
        y = super().forward(x_m)
        mask = ~(y > -torch.inf).detach()
        return MaskedTensor(y, mask=mask, compressed=x.compressed)


class Upsample(nn.Upsample):
    def forward(self, x):
        if not isinstance(x, MaskedTensor):
            return super().forward(x)

        y = super().forward(x.strip())
        mask_u = super().forward(x.mask.to(dtype=x.dtype))
        mask = ~(mask_u == 0.0).detach()
        return MaskedTensor(y, mask=mask, compressed=x.compressed)
