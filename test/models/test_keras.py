"""
Tests for the PyTorch NN backend.
"""
import tensorflow as tf
from tensorflow import keras

from quantnn.models.keras import QuantileLoss, CrossEntropyLoss
import numpy as np
from quantnn import (QRNN,
                     set_default_backend,
                     get_default_backend)

def test_quantile_loss():
    """
    Ensure that quantile loss corresponds to half of absolute error
    loss and that masking works as expected.
    """
    loss = QuantileLoss([0.5], mask=-1e3)

    y_pred = np.random.rand(10, 1, 10)
    y = np.random.rand(10, 1, 10)

    l = loss(y, y_pred)

    dy = (y_pred - y)
    l_ref = 0.5 * np.mean(np.abs(dy))

    assert np.isclose(l, l_ref)

    y_pred = np.random.rand(20, 1, 10)
    y_pred[10:] = -2e3
    y = np.random.rand(20, 1, 10)
    y[10:] = -2e3

    loss = QuantileLoss([0.5], mask=-1e3)
    l = loss(y, y_pred)
    l_ref = loss(y[:10], y_pred[:10])
    assert np.isclose(l, l_ref)

def test_cross_entropy_loss():
    """
    Test masking for cross entropy loss.

    Need to take into account that Keras, by default, expects channels along last axis.
    """
    y_pred = np.random.rand(10, 10, 10).astype(np.float32)
    y = np.ones((10, 10, 1), dtype=np.float32)
    bins = np.linspace(0, 1, 11)
    y[:, :, 0] = 5

    loss = CrossEntropyLoss(bins, mask=-1.0)
    ref = -y_pred[:, :, 5] + np.log(np.exp(y_pred).sum(-1))
    assert np.all(np.isclose(loss(y, y_pred),
                             ref.mean()))


    y[5:, :, :] = -1.0
    y[:, 5:, :] = -1.0
    ref = -y_pred[:5, :5, 5] + np.log(np.exp(y_pred[:5, :5, :]).sum(-1))
    assert np.all(np.isclose(loss(y, y_pred),
                             ref.mean()))

def test_training_with_dataloader():
    """
    Ensure that training with a pytorch dataloader works.
    """
    set_default_backend("keras")
    x = np.random.rand(1024, 16)
    y = np.random.rand(1024)

    batched_data = [
        {
            "x": x[i * 128: (i + 1) * 128],
            "y": y[i * 128: (i + 1) * 128],
        }
        for i in range(1024 // 128)
    ]

    qrnn = QRNN(np.linspace(0.05, 0.95, 10), n_inputs=x.shape[1])
    qrnn.train(batched_data, n_epochs=1)


def test_training_with_dict():
    """
    Ensure that training with batch objects as dicts works.
    """
    set_default_backend("keras")
    x = np.random.rand(1024, 16)
    y = np.random.rand(1024)

    batched_data = [
        {
            "x": x[i * 128: (i + 1) * 128],
            "y": y[i * 128: (i + 1) * 128],
        }
        for i in range(1024 // 128)
    ]

    qrnn = QRNN(np.linspace(0.05, 0.95, 10), n_inputs=x.shape[1])

    qrnn.train(batched_data, n_epochs=1)


def test_training_with_dict_and_keys():
    """
    Ensure that training with batch objects as dicts and provided keys
    argument works.
    """
    set_default_backend("keras")
    x = np.random.rand(1024, 16)
    y = np.random.rand(1024)

    batched_data = [
        {
            "x": x[i * 128: (i + 1) * 128],
            "x_2": x[i * 128: (i + 1) * 128],
            "y": y[i * 128: (i + 1) * 128],
        }
        for i in range(1024 // 128)
    ]

    qrnn = QRNN(np.linspace(0.05, 0.95, 10), n_inputs=x.shape[1])
    qrnn.train(batched_data, n_epochs=1, keys=("x", "y"))


def test_training_multiple_outputs():
    """
    Ensure that training with batch objects as dicts and provided keys
    argument works.
    """
    set_default_backend("keras")

    class MultipleOutputModel(keras.Model):
        def __init__(self):
            super().__init__()
            self.hidden = keras.layers.Dense(128, "relu", input_shape=(16,))
            self.head_1 = keras.layers.Dense(11, None)
            self.head_2 = keras.layers.Dense(11, None)

        def call(self, x):
            x = self.hidden(x)
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
            "x": x[i * 128: (i + 1) * 128],
            "y": {
                "y_1": y[i * 128: (i + 1) * 128],
                "y_2": y[i * 128: (i + 1) * 128]
            }
        }
        for i in range(1024 // 128)
    ]

    model = MultipleOutputModel()
    qrnn = QRNN(np.linspace(0.05, 0.95, 11), model=model)
    qrnn.train(batched_data, n_epochs=10, keys=("x", "y"))
