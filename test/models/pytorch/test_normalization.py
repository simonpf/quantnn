"""
Tests for the quantnn.models.pytorch.normalization module.
"""
import torch
import numpy as np

from quantnn.models.pytorch.normalization import (
    LayerNormFirst,
    GRN
)


def test_layer_norm_first():
    """
    Assert that layer norm with channels along first dimensions works.
    """
    norm = LayerNormFirst(16)
    x = torch.rand(10, 16, 24, 24)
    y = norm(x)

    mu = y.mean(1).detach().numpy()
    assert np.all(np.isclose(mu, 0.0, atol=1e-5))


def test_grn():
    """
    Assert that GRN works.
    """
    norm = GRN(16)
    x = torch.rand(10, 16, 24, 24)
    y = norm(x)
    x = x.detach().numpy()
    y = y.detach().numpy()
    assert np.all(np.isclose(x, y))
