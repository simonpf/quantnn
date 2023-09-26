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

    def __init__(self, n_channels, channel_dim=1, eps=1e-6):
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
