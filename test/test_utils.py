import numpy as np
import xarray as xr

from quantnn.utils import (apply,
                           serialize_dataset,
                           deserialize_dataset)

def test_apply():

    f1 = lambda x: 2 * x
    f2 = lambda x, y: x + y

    d = {i: i for i in range(5)}

    a = apply(f1, 1)
    b = apply(f2, 1, 1)
    assert a == a
    assert b == b

    d_a = apply(f1, d)
    d_b = apply(f2, d, d)
    for k in d:
        assert k == d_a[k] // 2
        assert k == d_b[k] // 2

def test_serialization():
    """
    Make sure that serialization of xarray datasets works.
    """
    dataset_ref = xr.Dataset({"x": (("a", "b"), np.ones((10, 10)))})
    b = serialize_dataset(dataset_ref)
    dataset = deserialize_dataset(b)

    assert np.all(np.isclose(dataset["x"].data,
                             dataset_ref["x"].data))


