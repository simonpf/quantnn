"""
Tests the QRNN implementation for all available backends.
"""
from quantnn import QRNN, set_backend, get_backend
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
        set_backend(backend)
        qrnn = QRNN(np.linspace(0.05, 0.95, 10),
                    input_dimensions=self.x_train.shape[1])
        qrnn.train((self.x_train, self.y_train), maximum_epochs=1)

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
    def test_qrnn_datasets(self, backend):
        """
        Provide data as dataset object instead of numpy arrays.
        """
        set_backend(backend)
        backend = get_backend(backend)
        data = backend.BatchedDataset((self.x_train, self.y_train), 256)
        qrnn = QRNN(np.linspace(0.05, 0.95, 10),
                    input_dimensions=self.x_train.shape[1])
        qrnn.train(data, maximum_epochs=1)

    @pytest.mark.parametrize("backend", backends)
    def test_save_qrnn(self, backend):
        """
        Test saving and loading of QRNNs.
        """
        set_backend(backend)
        qrnn = QRNN(np.linspace(0.05, 0.95, 10),
                    input_dimensions=self.x_train.shape[1])
        f = tempfile.NamedTemporaryFile()
        qrnn.save(f.name)
        qrnn_loaded = QRNN.load(f.name)

        x_pred = qrnn.predict(self.x_train)
        x_pred_loaded = qrnn.predict(self.x_train)

        if not type(x_pred) == np.ndarray:
            x_pred = x_pred.detach()

        assert np.allclose(x_pred, x_pred_loaded)
