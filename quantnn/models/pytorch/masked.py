"""
quantnn.models.pytorch.masked
=============================

Provides torch modules that support masked tensors.
"""
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


class Conv2d(nn.Conv2d):
    """
    Masked version of 2D convolution.
    """
    def __init__(
            self,
            channels_in: int,
            channels_out: int,
            kernel_size: int,
            **kwargs
    ):
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

        kwargs["padding_mode"] = "reflect"
        self.counter = nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size,
            **kwargs
        )
        for param in self.counter.parameters():
            param.requires_grad = False

        self.counter.weight.fill_(channels_out / self.counter.weight.data.numel())

        if self.has_bias:
            self.sep_bias = nn.Parameter(torch.tensor([0.0]))


    def forward(self, x):
        if not isinstance(x, MaskedTensor):
            return super().forward(x)

        mask = x.mask
        x_m = torch.where(mask, 0.0, x.strip())
        y = super().forward(x_m)

        valid_frac = self.counter(1.0 - mask.to(dtype=x.dtype))
        print(valid_frac.min(), valid_frac.max())
        mask_new = ~(valid_frac > 0.0).detach()
        valid_frac = torch.where(mask_new, 1.0, valid_frac)

        print("no scaling")
        #y = y / valid_frac.detach()
        y = torch.where(mask_new.detach(), y.detach(), y)
        if self.has_bias:
            y = y + self.sep_bias

        return MaskedTensor(y, mask=mask_new, compressed=x.compressed)



class BatchNorm2d(nn.BatchNorm2d):
    """
    Masked implementation of 2D batch norm.
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

        print("BN TRAINING : ", bn_training)

        is_masked = isinstance(x, MaskedTensor)
        # Special handling is only required during training.
        if bn_training and is_masked:
            compressed = x.compressed
            mask = x.mask

            x = torch.Tensor(x)

            n_valid = (~mask).sum((0, 2, 3))
            x_m = torch.where(mask, 0, x)
            mean = x_m.sum((0, 2, 3)) / n_valid
            var = (x_m ** 2).sum((0, 2, 3)) / n_valid - mean ** 2


            mean = torch.where(n_valid > 0, mean, self.running_mean).detach()
            var = torch.where(n_valid > 0, var, self.running_var).detach()


            y = (x - mean[..., None, None]) / (var + self.eps).sqrt()[..., None, None]
            y = torch.where(mask, 0.0, y)

            if self.weight is not None:
                y = self.weight[..., None, None] * y + self.bias[..., None, None]

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


class BatchNorm1d(nn.BatchNorm1d):
    """
    Masked implementation of 2D batch norm.
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

            mean = torch.where(n_valid > 0, mean, self.running_mean).detach()
            var = torch.where(n_valid > 0, var, self.running_var).detach()

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
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None
    ):
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


    def forward(self, x):
        if not isinstance(x, MaskedTensor):
            return super().forward(x)

        mask = x.mask
        x_m = torch.where(mask, 0.0, x)
        y = super().forward(x_m)

        valid_frac = self.counter(1.0 - mask.to(dtype=x.dtype))
        mask_new = ~(valid_frac > 0.0).detach()
        valid_frac = torch.where(mask_new, 1.0, valid_frac)

        #y = y / valid_frac
        y = torch.where(mask_new, y.detach(), y)
        if self.sep_bias is not None:
            y = y + self.sep_bias

        return MaskedTensor(y, mask=mask_new, compressed=x.compressed)



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
        cts = super().forward(1.0 - mask.to(dtype=x_m.dtype)).detach()
        new_mask = ~(cts > 0.0)
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
