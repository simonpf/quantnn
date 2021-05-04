import numpy as np
from quantnn.normalizer import Normalizer, MinMaxNormalizer

def test_normalizer_2d():
    """
    Checks that all feature indices that are not excluded have zero
    mean and unit std. dev.
    """
    x = np.random.normal(size=(100000, 10)) + np.arange(10).reshape(1, -1)
    normalizer = Normalizer(x,
                            exclude_indices=range(1, 10, 2))

    x_normed = normalizer(x)

    # Included indices should have zero mean and std. dev. 1.0.
    assert np.all(np.isclose(x_normed[:, ::2].mean(axis=0),
                             0.0,
                             atol=1e-1))
    assert np.all(np.isclose(x_normed[:, ::2].std(axis=0),
                             1.0,
                             1e-1))

    # Excluded indices
    assert np.all(np.isclose(x_normed[:, 1::2].mean(axis=0),
                             np.arange(10)[1::2].reshape(1, -1),
                             1e-2))
    assert np.all(np.isclose(x_normed[:, 1::2].std(axis=0),
                             1.0,
                             1e-2))

    # Channels without variation should be set to -1.0
    x = np.zeros((100, 10))
    normalizer = Normalizer(x)
    x_normed = normalizer(x)
    assert np.all(np.isclose(x_normed, -1.0))

def test_min_max_normalizer_2d():
    """
    Checks that all feature indices that are not excluded have zero
    mean and unit std. dev.
    """
    x = np.random.normal(size=(100000, 11)) + np.arange(11).reshape(1, -1)
    normalizer = MinMaxNormalizer(x, exclude_indices=range(1, 10, 2))
    x[:, 10] = np.nan

    x_normed = normalizer(x)

    # Included indices should have minimum value -0.9 and
    # maximum value 1.0.
    assert np.all(np.isclose(x_normed[:, :10:2].min(axis=0),
                             -1.0))
    assert np.all(np.isclose(x_normed[:, :10:2].max(axis=0),
                             1.0))
    # nan values should be set to -1.0.
    assert np.all(np.isclose(x_normed[:, -1], -1.5))

    # Channels without variation should be set to -1.0
    x = np.zeros((100, 10))
    normalizer = MinMaxNormalizer(x)
    x_normed = normalizer(x)
    assert np.all(np.isclose(x_normed, -1.0))

def test_invert():
    """
    Ensure that the inverse function of the Normalizer works as expected.
    """
    x = np.random.normal(size=(100000, 10)) + np.arange(10).reshape(1, -1)
    normalizer = Normalizer(x, exclude_indices=[0, 1, 2])

    x_normed = normalizer(x)
    x = normalizer.invert(x_normed)

    assert np.all(np.isclose(np.mean(x, axis=0),
                             np.arange(10, dtype=np.float32),
                             atol=1e-2))

def test_save_and_load(tmp_path):
    """
    Ensure that saved and loaded normalizer yields same results as original.
    """
    x = np.random.normal(size=(100000, 10)) + np.arange(10).reshape(1, -1)
    normalizer = Normalizer(x,
                            exclude_indices=range(1, 10, 2))
    normalizer.save(tmp_path / "normalizer.pckl")
    loaded = Normalizer.load(tmp_path / "normalizer.pckl")

    x_normed = normalizer(x)
    x_normed_loaded = loaded(x)

    assert np.all(np.isclose(x_normed,
                             x_normed_loaded))


def test_load_sftp(tmp_path):
    """
    Ensure that saved and loaded normalizer yields same results as original.
    """
    x = np.random.normal(size=(100000, 10)) + np.arange(10).reshape(1, -1)
    normalizer = Normalizer(x,
                            exclude_indices=range(1, 10, 2))
    normalizer.save(tmp_path / "normalizer.pckl")
    loaded = Normalizer.load(tmp_path / "normalizer.pckl")

    x_normed = normalizer(x)
    x_normed_loaded = loaded(x)

    assert np.all(np.isclose(x_normed,
                             x_normed_loaded,
                             rtol=1e-3))
