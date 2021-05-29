import numpy as np
import pytest
import scipy as sp

from quantnn.backends import TENSOR_BACKENDS

@pytest.mark.parametrize("backend", TENSOR_BACKENDS)
def test_conversion(backend):
    """
    Ensure that conversion back and forth from numpy arrays works.
    """

    x = np.random.rand(10, 10)
    x_b = backend.from_numpy(x)
    x_c = backend.to_numpy(x_b)

    assert np.all(np.isclose(x, x_c))

@pytest.mark.parametrize("backend", TENSOR_BACKENDS)
def test_sample_uniform(backend):
    x = backend.to_numpy(backend.sample_uniform((100, 100)))
    assert np.all(np.logical_and(x >= 0.0, x <= 1.0))

@pytest.mark.parametrize("backend", TENSOR_BACKENDS)
def test_sample_gaussian(backend):
    x = backend.to_numpy(backend.sample_gaussian((100, 100)))
    assert np.isclose(x.mean(), 0.0, atol=1e-1)
    assert np.isclose(x.std(), 1.0, atol=1e-1)

@pytest.mark.parametrize("backend", TENSOR_BACKENDS)
def test_size(backend):
    x = np.arange(10)
    x = backend.from_numpy(x)
    n = backend.size(x)
    assert n == 10

@pytest.mark.parametrize("backend", TENSOR_BACKENDS)
def test_concatenate(backend):
    x = np.arange(10)
    x = backend.from_numpy(x)
    xs = [backend.expand_dims(x, 0)] * 10
    xs = backend.concatenate(xs, 0)
    xs = backend.to_numpy(xs)

    for i in range(10):
        assert np.all(np.isclose(xs[:, i], i))

@pytest.mark.parametrize("backend", TENSOR_BACKENDS)
def test_expand_dims(backend):
    x = np.arange(10)
    x = backend.from_numpy(x)
    y = backend.expand_dims(x, 0)

    assert len(y.shape) == len(x.shape) + 1
    assert y.shape[0] == 1

@pytest.mark.parametrize("backend", TENSOR_BACKENDS)
def test_exp(backend):
    x = np.arange(10).astype(np.float)
    x = backend.from_numpy(x)
    y = backend.exp(x)

    assert np.all(np.isclose(backend.to_numpy(y), np.exp(x)))

@pytest.mark.parametrize("backend", TENSOR_BACKENDS)
def test_log(backend):
    x = np.arange(10).astype(np.float)
    x = backend.from_numpy(x)
    y = backend.log(x)

    assert np.all(np.isclose(backend.to_numpy(y), np.log(x)))


@pytest.mark.parametrize("backend", TENSOR_BACKENDS)
def test_pad_zeros(backend):
    x = np.arange(10)
    x = backend.from_numpy(x)
    xs = [backend.expand_dims(x, 0)] * 10
    xs = backend.concatenate(xs, 0)

    xs = backend.pad_zeros(xs, 2, 0)
    xs = backend.pad_zeros(xs, 1, 1)
    xs = backend.to_numpy(xs)


    assert np.all(xs[:2, :] == 0.0)
    assert np.all(xs[-2:, :] == 0.0)

    assert np.all(xs[:, :1] == 0.0)
    assert np.all(xs[:, -1:] == 0.0)

@pytest.mark.parametrize("backend", TENSOR_BACKENDS)
def test_pad_zeros_left(backend):
    x = np.arange(10)
    x = backend.from_numpy(x)
    xs = [backend.expand_dims(x, 0)] * 10
    xs = backend.concatenate(xs, 0)

    xs = backend.pad_zeros_left(xs, 2, 0)
    xs = backend.pad_zeros_left(xs, 1, 1)
    xs = backend.to_numpy(xs)

    assert np.all(xs[:2, :] == 0.0)
    assert not np.all(xs[-2:, :] == 0.0)

    assert np.all(xs[:, :1] == 0.0)
    assert not np.all(xs[:, -1:] == 0.0)

@pytest.mark.parametrize("backend", TENSOR_BACKENDS)
def test_arange(backend):
    x = backend.arange(2, 10, 2)
    x = backend.to_numpy(x)

    assert np.all(np.isclose(x, np.arange(2, 10, 2)))

@pytest.mark.parametrize("backend", TENSOR_BACKENDS)
def test_reshape(backend):
    x = backend.arange(2, 10, 2)
    x = backend.reshape(x, (-1, 1))
    x = backend.to_numpy(x)
    assert np.all(np.isclose(x, np.arange(2, 10, 2).reshape(-1, 1)))

@pytest.mark.parametrize("backend", TENSOR_BACKENDS)
def test_trapz(backend):

    x = backend.arange(2, 10, 2)
    y = backend.ones(like=x)

    integral = backend.to_numpy(backend.trapz(y, x, 0))
    assert np.all(np.isclose(integral, 6))

@pytest.mark.parametrize("backend", TENSOR_BACKENDS)
def test_cumsum(backend):

    x = backend.arange(2, 10, 2)
    cumsum = backend.to_numpy(backend.cumsum(x, 0))
    assert np.all(np.isclose(cumsum, np.cumsum(x)))

@pytest.mark.parametrize("backend", TENSOR_BACKENDS)
def test_zeros(backend):

    zeros_1 = backend.zeros((2, 2, 2))
    zeros_2 = backend.zeros(like=zeros_1)
    zeros_1 = backend.to_numpy(zeros_1)
    zeros_2 = backend.to_numpy(zeros_2)

    assert np.all(np.isclose(zeros_1, 0.0))
    assert np.all(np.isclose(zeros_1,
                             zeros_2))

@pytest.mark.parametrize("backend", TENSOR_BACKENDS)
def test_ones(backend):

    ones_1 = backend.ones((2, 2, 2))
    ones_2 = backend.ones(like=ones_1)
    ones_1 = backend.to_numpy(ones_1)
    ones_2 = backend.to_numpy(ones_2)

    assert np.all(np.isclose(ones_1, 1.0))
    assert np.all(np.isclose(ones_1,
                             ones_2))

@pytest.mark.parametrize("backend", TENSOR_BACKENDS)
def test_softmax(backend):

    x = backend.sample_uniform((10, 10))
    x_sf = backend.softmax(x, 1)
    x_np = backend.to_numpy(x)
    x_sf_np = sp.special.softmax(x_np, 1)

    assert np.all(np.isclose(x_sf, x_sf_np))

@pytest.mark.parametrize("backend", TENSOR_BACKENDS)
def test_where(backend):
    x = backend.ones((10, 10))
    y = backend.zeros((10, 10))
    z = backend.where(x > 0, y, x)
    z_np = backend.to_numpy(z)
    assert np.all(np.isclose(z_np, 0.0))

    x = backend.ones((10, 10))
    y = backend.zeros((10, 10))
    z = backend.where(x > 1, y, x)
    z_np = backend.to_numpy(z)
    assert np.all(np.isclose(z_np, 1.0))
