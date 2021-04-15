"""
Tests the QRNN implementation for all available backends.
"""
from quantnn import (QRNN,
                     set_default_backend,
                     get_default_backend)
import numpy as np
import os
import importlib
import pytest
import tempfile

#
# Import available backends.
#

backends = []
try:
    import quantnn.models.keras

    backends += ["keras"]
except:
    pass

try:
    import quantnn.models.pytorch

    backends += ["pytorch"]
except:
    pass


class TestQrnn:
    def setup_method(self):
        dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir, "test_data")
        x_train = np.load(os.path.join(path, "x_train.npy"))
        x_mean = np.mean(x_train, keepdims=True)
        x_sigma = np.std(x_train, keepdims=True)
        self.x_train = (x_train - x_mean) / x_sigma
        self.y_train = np.load(os.path.join(path, "y_train.npy"))

    @pytest.mark.parametrize("backend", backends)
    def test_qrnn(self, backend):
        """
        Test training of QRNNs using numpy arrays as input.
        """
        set_default_backend(backend)
        qrnn = QRNN(np.linspace(0.05, 0.95, 10),
                    n_inputs=self.x_train.shape[1])
        qrnn.train((self.x_train, self.y_train),
                   validation_data=(self.x_train, self.y_train),
                   n_epochs=2)

        qrnn.predict(self.x_train)

        x, qs = qrnn.cdf(self.x_train[:2, :])
        assert qs[0] == 0.0
        assert qs[-1] == 1.0

        x, y = qrnn.pdf(self.x_train[:2, :])
        assert x.shape == y.shape

        mu = qrnn.posterior_mean(self.x_train[:2, :])
        assert len(mu.shape) == 1

        r = qrnn.sample_posterior(self.x_train[:4, :], n_samples=2)
        assert r.shape == (4, 2)

        r = qrnn.sample_posterior_gaussian_fit(self.x_train[:4, :], n_samples=2)
        assert r.shape == (4, 2)

    @pytest.mark.parametrize("backend", backends)
    def test_qrnn_dict_iterable(self, backend):
        """
        Test training with dataset object that yields dicts instead of
        tuples.
        """
        set_default_backend(backend)
        backend = get_default_backend()

        class DictWrapper:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                for x, y in self.data:
                    yield {"x": x, "y": y}

            def __len__(self):
                return len(self.data)

        data = backend.BatchedDataset((self.x_train, self.y_train), 256)
        qrnn = QRNN(np.linspace(0.05, 0.95, 10),
                    n_inputs=self.x_train.shape[1])
        qrnn.train(DictWrapper(data), n_epochs=2, keys=("x", "y"))

    @pytest.mark.parametrize("backend", backends)
    def test_qrnn_datasets(self, backend):
        """
        Provide data as dataset object instead of numpy arrays.
        """
        set_default_backend(backend)
        backend = get_default_backend()
        data = backend.BatchedDataset((self.x_train, self.y_train), 256)
        qrnn = QRNN(np.linspace(0.05, 0.95, 10),
                    n_inputs=self.x_train.shape[1])
        qrnn.train(data, n_epochs=2)

    @pytest.mark.parametrize("backend", backends)
    def test_save_qrnn(self, backend):
        """
        Test saving and loading of QRNNs.
        """
        set_default_backend(backend)
        qrnn = QRNN(np.linspace(0.05, 0.95, 10),
                    n_inputs=self.x_train.shape[1])
        f = tempfile.NamedTemporaryFile()
        qrnn.save(f.name)
        qrnn_loaded = QRNN.load(f.name)

        x_pred = qrnn.predict(self.x_train)
        x_pred_loaded = qrnn.predict(self.x_train)

        if not type(x_pred) == np.ndarray:
            x_pred = x_pred.detach()

        assert np.allclose(x_pred, x_pred_loaded)

    @pytest.mark.skipif(not "pytorch" in backends,
                        reason="No PyTorch backend.")
    def test_save_qrnn_pytorch_model(self):
        """
        Test saving and loading of QRNNs.
        """
        from torch import nn
        quantiles = np.linspace(0.05, 0.95, 10)
        model = nn.Sequential(nn.Linear(self.x_train.shape[1], quantiles.size))
        qrnn = QRNN(quantiles, model=model)

        # Train the model
        data = quantnn.models.pytorch.BatchedDataset((self.x_train, self.y_train), 256)
        qrnn.train(data, n_epochs=2)

        # Save the model
        f = tempfile.NamedTemporaryFile()
        qrnn.save(f.name)
        qrnn_loaded = QRNN.load(f.name)

        # Compare predictions from saved and loaded model.
        x_pred = qrnn.predict(self.x_train)
        x_pred_loaded = qrnn.predict(self.x_train)
        if not type(x_pred) == np.ndarray:
            x_pred = x_pred.detach()

        assert np.allclose(x_pred, x_pred_loaded)
