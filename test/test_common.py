import numpy as np
import pytest
from qrnn.common import (get_array_module,
                         to_array,
                         sample_uniform)


@pytest.mark.parametrize("backend", pytest.backends)
def test_get_array_module(backend):
    """
    Ensures that get_array_module returns right array object
    when given an array created using the arange method of the
    corresponding module object.
    """
    x = backend.arange(10)
    module = get_array_module(x)
    assert module == backend

@pytest.mark.parametrize("backend", pytest.backends)
def test_to_array(backend):
    """
    Converts numpy array to array of given backend and ensures
    that corresponding module object matches the backend.
    """
    x = np.arange(10)
    array = to_array(backend, x)
    assert get_array_module(array) == backend

@pytest.mark.parametrize("backend", pytest.backends)
def test_uniform_sample(backend):
    """
    Ensures that array of random samples has array type
    corresponding to the right backend module.
    """
    samples = sample_uniform(backend, (10, ))
    assert get_array_module(samples) == backend
