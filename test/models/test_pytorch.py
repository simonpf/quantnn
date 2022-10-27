"""
Tests for the PyTorch NN backend.
"""
import torch
from torch import nn
import numpy as np

from quantnn import set_default_backend
from quantnn.qrnn import QRNN
from quantnn.drnn import DRNN, _to_categorical
from quantnn.mrnn import Quantiles, Density, Mean, Classification, MRNN

from quantnn.models.pytorch import QuantileLoss, CrossEntropyLoss, MSELoss
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

    # Test loss for multi-class classification.
    y_pred = torch.rand(10, 10, 10)
    y = torch.ones(10, 10, dtype=torch.long)
    y[:, :] = 5

    loss = CrossEntropyLoss(10, mask=-1.0)
    ref = -y_pred[:, 5, :] + torch.log(torch.exp(y_pred).sum(1))
    assert np.all(np.isclose(loss(y_pred, y).detach().numpy(),
                             ref.mean().detach().numpy()))


    y[5:, :] = -1.0
    y[:, 5:] = -1.0
    ref = -y_pred[:5, 5, :5] + torch.log(torch.exp(y_pred[:5, :, :5]).sum(1))
    assert np.all(np.isclose(loss(y_pred, y).detach().numpy(),
                             ref.mean().detach().numpy()))

    # Test loss for binary classification.
    y_pred = torch.rand(10, 1, 10)
    y = torch.ones(10, 1, 10, dtype=torch.long)
    y[:, :] = 1

    loss = CrossEntropyLoss(2, mask=-1)
    ref = -torch.nn.functional.logsigmoid(y_pred)
    assert np.all(np.isclose(loss(y_pred, y).detach().numpy(),
                             ref.mean().detach().numpy()))


    y[5:, :] = -1.0
    y[:, 5:] = -1.0
    ref = - torch.nn.functional.logsigmoid(y_pred[:5, :5])
    assert np.all(np.isclose(loss(y_pred, y).detach().numpy(),
                             ref.mean().detach().numpy()))



def test_mse_loss():
    """
    Test masking for cross entropy loss.
    """
    set_default_backend("pytorch")

    y_pred = torch.rand(10, 10, 10)
    y = torch.ones(10, 10, 10)
    y[:, 0, :] = 0.55

    loss = MSELoss(mask=-1.0)
    ref = ((y_pred - y) ** 2).mean()
    assert np.all(np.isclose(loss(y_pred, y).detach().numpy(),
                             ref.mean().detach().numpy()))


    y[5:, :, :] = -1.0
    y[:, :, 5:] = -1.0
    ref = ((y_pred[:5, :, :5] - y[:5, :, :5]) ** 2).mean()
    assert np.all(np.isclose(loss(y_pred, y).detach().numpy(),
                             ref.mean().detach().numpy()))


def test_qrnn_training_state():
    """
    Ensure that training attributes of models are conserved through
    training.
    """
    set_default_backend("pytorch")

    x = np.random.rand(1024, 16)
    y = np.random.rand(1024)
    training_data = torch.utils.data.TensorDataset(torch.tensor(x),
                                                   torch.tensor(y))
    training_loader = torch.utils.data.DataLoader(training_data, batch_size=128)

    model = nn.Sequential(
        nn.BatchNorm1d(16),
        nn.Linear(16, 10)
    )
    qrnn = QRNN(np.linspace(0.05, 0.95, 10), model=model)

    qrnn.model.train(False)
    qrnn.train(training_loader, n_epochs=1)

    mean = model[0].running_mean.detach().numpy()
    assert np.all(np.isclose(mean, 0.0))
    var = model[0].running_var.detach().numpy()
    assert np.all(np.isclose(var, 1.0))

    qrnn.model.train(True)
    qrnn.train(training_loader, n_epochs=1)

    mean = model[0].running_mean.detach().numpy()
    assert not np.all(np.isclose(mean, 0.0))
    var = model[0].running_var.detach().numpy()
    assert not np.all(np.isclose(var, 1.0))


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
    metrics = ["Bias", "MeanSquaredError", "CRPS"]
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
    metrics = ["Bias", "MeanSquaredError", "CRPS"]
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
    bins = np.linspace(0, 1, 12)
    bins = {"y_1": bins, "y_2": bins}
    qrnn = DRNN(bins=bins, model=model)
    metrics = ["Bias", "MeanSquaredError", "CRPS", "ScatterPlot", "QuantileFunction"]
    qrnn.train(batched_data,
               validation_data=batched_data,
               n_epochs=5, keys=("x", "y"),
               metrics=metrics)


def test_training_multi_mrnn():
    """
    Ensure that training with batch objects as dicts and provided keys
    argument works.
    """
    set_default_backend("pytorch")

    class MultipleOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(16, 128)
            self.head_1 = nn.Linear(128, 10)
            self.head_2 = nn.Linear(128, 20)
            self.head_3 = nn.Linear(128, 1)
            self.head_4 = nn.Linear(128, 1)
            self.head_5 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.hidden(x))
            y_1 = self.head_1(x)
            y_2 = self.head_2(x)
            y_3 = self.head_3(x)
            y_4 = self.head_4(x)
            y_5 = self.head_5(x)
            return {
                "quantiles": y_1,
                "density": y_2,
                "mean": y_3,
                "binary_classification": y_4,
                "classification": y_5,
            }

    x = np.random.rand(2024, 16) + 1.0
    y = np.sum(x, axis=-1)
    y += np.random.normal(size=y.size)

    batched_data = []
    for i in range(512):
        x = torch.as_tensor(np.random.normal(size=(128, 16)))
        r = torch.as_tensor(np.random.normal(size=128), dtype=torch.float)
        y = {
            "quantiles": r,
            "density": r,
            "mean": r,
            "binary_classification": torch.as_tensor(r > 0, dtype=torch.long),
            "classification": torch.as_tensor(
                np.digitize(r, np.linspace(-1, 1, 11)) - 2,
                dtype=torch.long
            ),
        }
        batched_data.append((x, y))

    model = MultipleOutputModel()

    bins = np.linspace(0, 1, 12)
    bins = {"y_1": bins, "y_2": bins}

    losses = {
        "quantiles": Quantiles(np.linspace(0.05, 0.95, 10)),
        "density": Density(np.linspace(-2, 2, 21)),
        "mean": Mean(),
        "binary_classification": Classification(2),
        "classification": Classification(10)
    }

    metrics = [
        "Bias",
        "CRPS",
        "MeanSquaredError",
        "ScatterPlot",
        "CalibrationPlot",
        "Correlation"
    ]

    mrnn = MRNN(losses=losses, model=model)
    mrnn.train(
        batched_data,
        validation_data=batched_data[:10],
        n_epochs=4,
        metrics=metrics
    )

    x = batched_data[0][0].to(torch.float)
    with torch.no_grad():
        y_pred = mrnn.model(x)
        y_binary = torch.nn.functional.sigmoid(y_pred["binary_classification"])
        assert np.all(np.isclose(y_binary.numpy(), 0.5, atol=0.10))


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
    metrics = [
        "Bias",
        "CRPS",
        "MeanSquaredError",
        "ScatterPlot",
        "CalibrationPlot",
        "Correlation"
    ]
    qrnn.train(batched_data,
               validation_data=batched_data,
               n_epochs=5, keys=("x", "y"),
               metrics=metrics)


def test_training_transformation_mrnn_quantiles():
    """
    Ensure that training in transformed space works.
    """
    set_default_backend("pytorch")

    class MultipleOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(16, 128)
            self.head_1 = nn.Linear(128, 10)
            self.head_2 = nn.Linear(128, 1)
            self.head_3 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.hidden(x))
            y_1 = self.head_1(x)
            y_2 = self.head_2(x)
            y_3 = self.head_3(x)
            return {
                "y_1": y_1,
                "y_2": y_2,
                "y_3": y_3
            }

    x = np.random.rand(2024, 16) + 1.0
    y = np.sum(x, axis=-1)
    y += np.random.normal(size=y.size)

    batched_data = [
        {
            "x": torch.tensor(x[i * 128: (i + 1) * 128],
                              dtype=torch.float32),
            "y": {
                "y_1": torch.tensor(y[i * 128: (i + 1) * 128],
                                    dtype=torch.float32),
                "y_2": torch.tensor(y[i * 128: (i + 1) * 128] ** 2,
                                    dtype=torch.float32),
                "y_3": torch.tensor(y[i * 128: (i + 1) * 128] ** 2,
                                    dtype=torch.float32)
            }
        }
        for i in range(1024 // 128)
    ]

    model = MultipleOutputModel()
    transformations = {
        "y_1": Log10(),
        "y_2": Log10()
    }
    losses = {
        "y_1": Quantiles(np.linspace(0.05, 0.95, 10)),
        "y_2": Mean(),
        "y_3": Density(np.linspace(-2, 2, 11))
    }

    mrnn = MRNN(losses=losses, model=model)
    metrics = [
        "Bias",
        "CRPS",
        "MeanSquaredError",
        "ScatterPlot",
        "CalibrationPlot",
        "Correlation"
    ]
    mrnn.train(batched_data,
               validation_data=batched_data,
               n_epochs=5, keys=("x", "y"),
               metrics=metrics)


def test_training_transformation_mrnn_density():
    """
    Ensure that training in transformed space works.
    """
    set_default_backend("pytorch")

    class MultipleOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(16, 128)
            self.head_1 = nn.Linear(128, 10)
            self.head_2 = nn.Linear(128, 1)

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
            "x": torch.tensor(x[i * 128: (i + 1) * 128],
                              dtype=torch.float32),
            "y": {
                "y_1": torch.tensor(y[i * 128: (i + 1) * 128],
                                    dtype=torch.float32),
                "y_2": torch.tensor(y[i * 128: (i + 1) * 128] ** 2,
                                    dtype=torch.float32)
            }
        }
        for i in range(1024 // 128)
    ]

    model = MultipleOutputModel()
    transformations = {
        "y_1": Log10(),
        "y_2": Log10()
    }
    losses = {
        "y_1": Density(np.linspace(0.05, 0.95, 11)),
        "y_2": Mean()
    }

    mrnn = MRNN(losses=losses, model=model)
    metrics = [
        "Bias",
        "CRPS",
        "MeanSquaredError",
        "ScatterPlot",
        "CalibrationPlot",
        "Correlation"
    ]
    mrnn.train(batched_data,
               validation_data=batched_data,
               n_epochs=5, keys=("x", "y"),
               metrics=metrics)


def test_qrnn_training_metrics_conv():
    """
    E

    """
    set_default_backend("pytorch")

    x_train = np.random.rand(1024, 16, 32, 32,)
    y_train = np.random.rand(1024, 1, 32, 32)
    x_val = np.random.rand(32, 16, 32, 32,)
    y_val = np.random.rand(32, 1, 32, 32)

    training_data = torch.utils.data.TensorDataset(torch.tensor(x_train),
                                                   torch.tensor(y_train))
    training_loader = torch.utils.data.DataLoader(training_data, batch_size=128)
    validation_data = torch.utils.data.TensorDataset(torch.tensor(x_val),
                                                     torch.tensor(y_val))
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=1)

    model = nn.Sequential(
        nn.Conv2d(16, 10, 1)
    )

    qrnn = QRNN(np.linspace(0.05, 0.95, 10), model=model)

    metrics = [
        "Bias",
        "MeanSquaredError",
        "CRPS",
        "CalibrationPlot",
        "ScatterPlot",
        "Correlation"
    ]
    qrnn.train(training_loader,
               validation_data=validation_loader,
               n_epochs=2,
               metrics=metrics,
               batch_size=1,
               mask=-1)
