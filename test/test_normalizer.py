import numpy as np
from quantnn.normalizer import Normalizer

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
                             atol=1e-2))
    assert np.all(np.isclose(x_normed[:, ::2].std(axis=0),
                             1.0,
                             1e-2))

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

    
