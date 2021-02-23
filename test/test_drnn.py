"""
Tests for quantnn.drnn module.
"""
import numpy as np

from quantnn import drnn

def test_to_categorical():
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

