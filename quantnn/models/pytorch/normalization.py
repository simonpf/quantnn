"""
quantnn.models.pytorch.normalization
====================================

This module implements normalization layers.
"""
import torch
from torch import nn


class LayerNormFirst(nn.Module):
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
        super().__init__()
        self.scaling = nn.Parameter(torch.ones(n_channels))
        self.bias = nn.Parameter(torch.zeros(n_channels))
        self.eps = eps

    def forward(self, x):
        """
        Apply normalization to x.
        """
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x_n = (x - mu) / torch.sqrt(var + self.eps)
        x = self.scaling[..., None, None] * x_n + self.bias[..., None, None]
        return x


class GRN(nn.Module):
    """
    Global Response Normalization (GRN) as proposed in https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html
    """
    def __init__(self, n_channels, eps=1e-6):
        """
        n_channels: Number of channels over which to normalize.
        eps: Epsilon added to mean to avoid numerical issues.
        """
        super().__init__()
        self.scaling = nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, n_channels, 1, 1))

    def forward(self, x):
        """
        Apply normalization to x.
        """
        x_l2 = torch.norm(x, p=2, dim=(-2, -1), keepdim=True)
        rel_imp = x_l2 / (x_l2.mean(dim=1, keepdim=True) + 1e-6)
        return self.scaling * (x * rel_imp) + self.bias + x
