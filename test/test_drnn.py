"""
Tests for quantnn.drnn module.
"""
import os
import tempfile

import pytest

import numpy as np
from quantnn import drnn, set_default_backend, get_default_backend
from quantnn.drnn import DRNN

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

class TestDrnn:
    def setup_method(self):
        dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir, "test_data")
        x_train = np.load(os.path.join(path, "x_train.npy"))
        x_mean = np.mean(x_train, keepdims=True)
        x_sigma = np.std(x_train, keepdims=True)
        self.x_train = (x_train - x_mean) / x_sigma
        self.bins = np.logspace(0, 3, 21)
        y = np.load(os.path.join(path, "y_train.npy"))
        self.y_train = y

    def test_to_categorical(self):
        """
        Assert that converting a continuous target variable to binned
        representation works as expected.
        """
        bins = np.linspace(0, 10, 11)

        y = np.arange(12) - 0.5
        y_cat = drnn._to_categorical(y, bins)

        assert y_cat[0] == 0
        assert np.all(np.isclose(y_cat[1:-1], np.arange(10)))
        assert y_cat[-1] == 9

    @pytest.mark.parametrize("backend", backends)
    def test_drnn(self, backend):
        """
        Test training of DRNN using numpy arrays as input.
        """
        set_default_backend(backend)
        drnn = DRNN(self.bins,
                    n_inputs=self.x_train.shape[1])
        drnn.train((self.x_train, self.y_train),
                   validation_data=(self.x_train, self.y_train),
                   n_epochs=2)

        drnn.predict(self.x_train)

        mu = drnn.posterior_mean(self.x_train[:2, :])
        assert len(mu.shape) == 1

        r = drnn.sample_posterior(self.x_train[:4, :], n_samples=2)
        assert r.shape == (4, 2)

    @pytest.mark.parametrize("backend", backends)
    def test_drnn_dict_iterable(self, backend):
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
        drnn = DRNN(self.bins,
                    n_inputs=self.x_train.shape[1])
        drnn.train(DictWrapper(data), n_epochs=2, keys=("x", "y"))

    @pytest.mark.parametrize("backend", backends)
    def test_drnn_datasets(self, backend):
        """
        Provide data as dataset object instead of numpy arrays.
        """
        set_default_backend(backend)
        backend = get_default_backend()
        data = backend.BatchedDataset((self.x_train, self.y_train), 256)
        drnn = DRNN(self.bins,
                    n_inputs=self.x_train.shape[1])
        drnn.train(data, n_epochs=2)

    @pytest.mark.parametrize("backend", backends)
    def test_save_drnn(self, backend):
        """
        Test saving and loading of DRNNs.
        """
        set_default_backend(backend)
        drnn = DRNN(self.bins,
                    n_inputs=self.x_train.shape[1])
        f = tempfile.NamedTemporaryFile()
        drnn.save(f.name)
        drnn_loaded = DRNN.load(f.name)

        x_pred = drnn.predict(self.x_train)
        x_pred_loaded = drnn.predict(self.x_train)

        if not type(x_pred) == np.ndarray:
            x_pred = x_pred.detach()

        assert np.allclose(x_pred, x_pred_loaded)
