"""
Tests for the PyTorch NN backend.
"""
from quantnn.models.pytorch import QuantileLoss, CrossEntropyLoss
import torch
import numpy as np

def test_quantile_loss():
    """
    Ensure that quantile loss corresponds to half of absolute error
    loss and that masking works as expected.
    """
    loss = QuantileLoss([0.5], mask=-1e3)

    y_pred = torch.rand(10, 1, 10)
    y = torch.rand(10, 1, 10)

    l = loss(y_pred, y).detach().numpy()

    dy = (y_pred - y).detach().numpy()
    l_ref = 0.5 * np.mean(np.abs(dy))

    assert np.isclose(l, l_ref)

    y_pred = torch.rand(20, 1, 10)
    y_pred[10:] = -2e3
    y = torch.rand(20, 1, 10)
    y[10:] = -2e3

    loss = QuantileLoss([0.5], mask=-1e3)
    l = loss(y_pred, y).detach().numpy()
    l_ref = loss(y_pred[:10], y[:10]).detach().numpy()
    assert np.isclose(l, l_ref)

def test_cross_entropy_loss():
    """
    Test masking for cross entropy loss.
    """
    y_pred = torch.rand(10, 10, 10)
    y = torch.ones(10, 1, 10, dtype=torch.long)
    y[:, 0, :] = 5

    loss = CrossEntropyLoss(mask=-1.0)
    ref = -y_pred[:, 5, :] + torch.log(torch.exp(y_pred).sum(1))
    assert np.all(np.isclose(loss(y_pred, y).detach().numpy(),
                             ref.mean().detach().numpy()))


    y[5:, :, :] = -1.0
    y[:, :, 5:] = -1.0
    ref = -y_pred[:5, 5, :5] + torch.log(torch.exp(y_pred[:5, :, :5]).sum(1))
    assert np.all(np.isclose(loss(y_pred, y).detach().numpy(),
                             ref.mean().detach().numpy()))




