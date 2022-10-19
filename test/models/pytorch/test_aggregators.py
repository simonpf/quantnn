import numpy as np
import pytest
import torch
from quantnn.packed_tensor import PackedTensor
from quantnn.models.pytorch.aggregators import (
    AverageAggregatorFactory,
    BlockAggregatorFactory,
    SparseAggregator,
    SumAggregatorFactory,
    LinearAggregatorFactory,
)
from quantnn.models.pytorch.torchvision import ResNetBlockFactory


def fill_tensor(t, indices):
    """
    Fills tensor with corresponding sample indices.
    """
    for i, ind in enumerate(indices):
        t[i] = ind
    return t


def make_random_packed_tensor(batch_size, samples, shape):
    """
    Create a sparse tensor representing a training batch with
    missing samples. Which samples are missing is randomized.
    The elements of the tensor correspond to the sample index.

    Args:
        batch_size: The nominal batch size of the training batch.
        samples: The number of non-missing samples.
        shape: The of each sample in the training batch.
    """
    indices = sorted(
        np.random.choice(np.arange(batch_size), size=samples, replace=False)
    )
    t = np.ones((samples,) + shape, dtype=np.float32)
    t = fill_tensor(t, indices)
    return PackedTensor(t, batch_size, indices)


def test_sparse_aggregator():
    """
    Tests the sparse aggregator by ensuring that sparse inputs are correctly
    combined.
    """
    aggregator_factory = AverageAggregatorFactory()
    aggregator = SparseAggregator(8, aggregator_factory)

    # Ensure full tensor is returned if only one of the provided
    # tensors is sparse.
    x_1 = torch.ones((100, 8, 32, 32), dtype=torch.float32)
    fill_tensor(x_1, range(100))
    x_2 = make_random_packed_tensor(100, 50, (8, 32, 32))
    y = aggregator(x_1, x_2)
    assert not isinstance(y, PackedTensor)
    assert torch.isclose(y, x_1).all()

    x_2 = torch.ones((100, 8, 32, 32), dtype=torch.float32)
    fill_tensor(x_2, range(100))
    x_1 = make_random_packed_tensor(100, 50, (8, 32, 32))
    y = aggregator(x_1, x_2)
    assert not isinstance(y, PackedTensor)
    assert torch.isclose(y, x_2).all()

    # Make sure merging works with two packed tensors.
    x_1 = make_random_packed_tensor(100, 50, (8, 32, 32))
    x_2 = make_random_packed_tensor(100, 50, (8, 32, 32))
    y = aggregator(x_1, x_2)
    assert isinstance(y, PackedTensor)
    batch_indices_union = sorted(list(set(x_1.batch_indices + x_2.batch_indices)))
    assert y.batch_indices == batch_indices_union
    for ind, batch_ind in enumerate(y.batch_indices):
        assert torch.isclose(y._t[ind], batch_ind * torch.ones(1, 1, 1)).all()

    # Test aggregation with linear layer.
    aggregator_factory = LinearAggregatorFactory()
    aggregator = SparseAggregator(8, aggregator_factory)

    # Ensure full tensor is returned if only one of the provided
    # tensors is sparse.
    x_1 = torch.ones((100, 8, 32, 32), dtype=torch.float32)
    fill_tensor(x_1, range(100))
    x_2 = make_random_packed_tensor(100, 50, (8, 32, 32))
    y = aggregator(x_1, x_2)
    assert not isinstance(y, PackedTensor)

    x_2 = torch.ones((100, 8, 32, 32), dtype=torch.float32)
    fill_tensor(x_2, range(100))
    x_1 = make_random_packed_tensor(100, 50, (8, 32, 32))
    y = aggregator(x_1, x_2)
    assert not isinstance(y, PackedTensor)

    # Make sure merging works with two packed tensors.
    x_1 = make_random_packed_tensor(100, 50, (8, 32, 32))
    x_2 = make_random_packed_tensor(100, 50, (8, 32, 32))
    y = aggregator(x_1, x_2)
    assert isinstance(y, PackedTensor)
    batch_indices_union = sorted(list(set(x_1.batch_indices + x_2.batch_indices)))
    assert y.batch_indices == batch_indices_union


AGGREGATORS = [
    SumAggregatorFactory(),
    AverageAggregatorFactory(),
    LinearAggregatorFactory(),
    BlockAggregatorFactory(ResNetBlockFactory()),
]


@pytest.mark.parametrize("aggregator", AGGREGATORS)
def test_aggregators(aggregator):
    a = torch.ones(1, 10, 16, 16)
    b = torch.ones(1, 10, 16, 16)
    agg = aggregator(10, 2, 10)
    c = agg(a, b)
    assert c.shape == (1, 10, 16, 16)

    c = torch.ones(1, 10, 16, 16)
    agg = aggregator(10, 3, 10)
    c = agg(a, b, c)
    assert c.shape == (1, 10, 16, 16)
