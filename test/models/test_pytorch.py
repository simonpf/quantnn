"""
Tests for the PyTorch NN backend.
"""
import torch
from torch import nn
import numpy as np

from quantnn import set_default_backend
from quantnn.qrnn import QRNN
from quantnn.drnn import DRNN, _to_categorical
from quantnn.models.pytorch import QuantileLoss, CrossEntropyLoss
from quantnn.transformations import Log10

def test_quantile_loss():
    """
    Ensure that quantile loss corresponds to half of absolute error
    loss and that masking works as expected.
    """
    set_default_backend("pytorch")

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
    set_default_backend("pytorch")

    y_pred = torch.rand(10, 10, 10)
    y = torch.ones(10, 1, 10)
    bins = np.linspace(0, 1, 11)
    y[:, 0, :] = 0.55

    loss = CrossEntropyLoss(bins, mask=-1.0)
    ref = -y_pred[:, 5, :] + torch.log(torch.exp(y_pred).sum(1))
    assert np.all(np.isclose(loss(y_pred, y).detach().numpy(),
                             ref.mean().detach().numpy()))


    y[5:, :, :] = -1.0
    y[:, :, 5:] = -1.0
    ref = -y_pred[:5, 5, :5] + torch.log(torch.exp(y_pred[:5, :, :5]).sum(1))
    assert np.all(np.isclose(loss(y_pred, y).detach().numpy(),
                             ref.mean().detach().numpy()))

def test_qrnn_training_with_dataloader():
    """
    Ensure that training with a pytorch dataloader works.
    """
    set_default_backend("pytorch")

    x = np.random.rand(1024, 16)
    y = np.random.rand(1024)

    training_data = torch.utils.data.TensorDataset(torch.tensor(x),
                                                   torch.tensor(y))
    training_loader = torch.utils.data.DataLoader(training_data, batch_size=128)
    qrnn = QRNN(np.linspace(0.05, 0.95, 10), n_inputs=x.shape[1])

    qrnn.train(training_loader, n_epochs=1)


def test_qrnn_training_with_dict():
    """
    Ensure that training with batch objects as dicts works.
    """
    set_default_backend("pytorch")

    x = np.random.rand(1024, 16)
    y = np.random.rand(1024)

    batched_data = [
        {
            "x": torch.tensor(x[i * 128: (i + 1) * 128]),
            "y": torch.tensor(y[i * 128: (i + 1) * 128]),
        }
        for i in range(1024 // 128)
    ]

    qrnn = QRNN(np.linspace(0.05, 0.95, 10), n_inputs=x.shape[1])

    qrnn.train(batched_data, n_epochs=1)


def test_qrnn_training_with_dict_and_keys():
    """
    Ensure that training with batch objects as dicts and provided keys
    argument works.
    """
    set_default_backend("pytorch")

    x = np.random.rand(1024, 16)
    y = np.random.rand(1024)

    batched_data = [
        {
            "x": torch.tensor(x[i * 128: (i + 1) * 128]),
            "x_2": torch.tensor(x[i * 128: (i + 1) * 128]),
            "y": torch.tensor(y[i * 128: (i + 1) * 128]),
        }
        for i in range(1024 // 128)
    ]

    qrnn = QRNN(np.linspace(0.05, 0.95, 10), n_inputs=x.shape[1])
    qrnn.train(batched_data, n_epochs=1, keys=("x", "y"))

def test_qrnn_training_metrics():
    """
    Ensure that training with a single target and metrics works.
    """
    set_default_backend("pytorch")

    x = np.random.rand(1024, 16)
    y = np.random.rand(1024)

    batched_data = [
        {
            "x": torch.tensor(x[i * 128: (i + 1) * 128]),
            "x_2": torch.tensor(x[i * 128: (i + 1) * 128]),
            "y": torch.tensor(y[i * 128: (i + 1) * 128]),
        }
        for i in range(1024 // 128)
    ]

    qrnn = QRNN(np.linspace(0.05, 0.95, 10), n_inputs=x.shape[1])
    metrics = ["Bias", "MeanSquaredError"]
    qrnn.train(batched_data, n_epochs=1, keys=("x", "y"), metrics=metrics)

def test_drnn_training_metrics():
    """
    Ensure that training with a single target and metrics works.
    """
    set_default_backend("pytorch")

    x = np.random.rand(1024, 16)
    bins = np.arange(128 * 8)
    y = _to_categorical(np.random.rand(1024), bins)

    batched_data = [
        {
            "x": torch.tensor(x[i * 128: (i + 1) * 128]),
            "x_2": torch.tensor(x[i * 128: (i + 1) * 128]),
            "y": torch.tensor(y[i * 128: (i + 1) * 128]),
        }
        for i in range(1024 // 128)
    ]

    drnn = DRNN(np.linspace(0.05, 0.95, 10), n_inputs=x.shape[1])
    metrics = ["Bias", "MeanSquaredError"]
    drnn.train(batched_data, n_epochs=1, keys=("x", "y"), metrics=metrics)


def test_training_multiple_outputs():
    """
    Ensure that training with batch objects as dicts and provided keys
    argument works.
    """
    set_default_backend("pytorch")

    class MultipleOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(16, 128)
            self.head_1 = nn.Linear(128, 11)
            self.head_2 = nn.Linear(128, 11)

        def forward(self, x):
            x = torch.relu(self.hidden(x))
            y_1 = self.head_1(x)
            y_2 = self.head_2(x)
            return {
                "y_1": y_1,
                "y_2": y_2
            }

    x = np.random.rand(1024, 16)
    y = np.random.rand(1024)

    batched_data = [
        {
            "x": torch.tensor(x[i * 128: (i + 1) * 128]),
            "y": {
                "y_1": torch.tensor(y[i * 128: (i + 1) * 128]),
                "y_2": torch.tensor(y[i * 128: (i + 1) * 128])
            }
        }
        for i in range(1024 // 128)
    ]

    model = MultipleOutputModel()
    qrnn = QRNN(np.linspace(0.05, 0.95, 11), model=model)
    qrnn.train(batched_data, n_epochs=5, keys=("x", "y"))

def test_training_metrics_multi():
    """
    Ensure that training with batch objects as dicts and provided keys
    argument works.
    """
    set_default_backend("pytorch")

    class MultipleOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(16, 128)
            self.head_1 = nn.Linear(128, 11)
            self.head_2 = nn.Linear(128, 11)

        def forward(self, x):
            x = torch.relu(self.hidden(x))
            y_1 = self.head_1(x)
            y_2 = self.head_2(x)
            return {
                "y_1": y_1,
                "y_2": y_2
            }

    x = np.random.rand(2024, 16) + 1.0
    y = np.sum(x, axis=-1)
    y += np.random.normal(size=y.size)

    batched_data = [
        {
            "x": torch.tensor(x[i * 128: (i + 1) * 128]),
            "y": {
                "y_1": torch.tensor(y[i * 128: (i + 1) * 128]),
                "y_2": torch.tensor(y[i * 128: (i + 1) * 128] ** 2)
            }
        }
        for i in range(1024 // 128)
    ]

    model = MultipleOutputModel()
    qrnn = QRNN(np.linspace(0.05, 0.95, 11), model=model)
    metrics = ["Bias", "CRPS", "MeanSquaredError", "ScatterPlot", "CalibrationPlot"]
    qrnn.train(batched_data,
               validation_data=batched_data,
               n_epochs=5, keys=("x", "y"),
               metrics=metrics)

def test_training_transformation():
    """
    Ensure that training in transformed space works.
    """
    set_default_backend("pytorch")

    class MultipleOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(16, 128)
            self.head_1 = nn.Linear(128, 11)
            self.head_2 = nn.Linear(128, 11)

        def forward(self, x):
            x = torch.relu(self.hidden(x))
            y_1 = self.head_1(x)
            y_2 = self.head_2(x)
            return {
                "y_1": y_1,
                "y_2": y_2
            }

    x = np.random.rand(2024, 16) + 1.0
    y = np.sum(x, axis=-1)
    y += np.random.normal(size=y.size)

    batched_data = [
        {
            "x": torch.tensor(x[i * 128: (i + 1) * 128]),
            "y": {
                "y_1": torch.tensor(y[i * 128: (i + 1) * 128]),
                "y_2": torch.tensor(y[i * 128: (i + 1) * 128] ** 2)
            }
        }
        for i in range(1024 // 128)
    ]

    model = MultipleOutputModel()
    transformations = {
        "y_1": Log10(),
        "y_2": None
    }
    qrnn = QRNN(np.linspace(0.05, 0.95, 11), model=model,
                transformation=transformations)
    metrics = ["Bias", "CRPS", "MeanSquaredError", "ScatterPlot", "CalibrationPlot"]
    qrnn.train(batched_data,
               validation_data=batched_data,
               n_epochs=5, keys=("x", "y"),
               metrics=metrics)
