import numpy as np
import torch
from torch import nn
from quantnn.packed_tensor import PackedTensor

def fill_tensor(t, indices):
    """
    Fills tensor with corresponding sample indices.
    """
    for i, ind in enumerate(indices):
        t[i] = ind
    return t


def make_random_packed_tensor(batch_size, samples, shape=(1,)):
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


def test_attributes():
    """
    Test attributes of packed tensor.
    """
    t = torch.ones((2, 2))
    t_p = PackedTensor(t, 4, [0, 1])

    assert t_p.batch_size == 4
    assert t_p.batch_indices == [0, 1]
    assert t_p.shape == (2, 2)


def test_stack():
    """
    Test stacking of list of tensor into batch.
    """
    t = torch.ones((2, 2))
    u = 2.0 * torch.ones((2, 2))

    tensors = [None, t, u, None]
    b = PackedTensor.stack(tensors)

    assert b.batch_indices == [1, 2]
    assert b.batch_size == 4

    b_e = b.expand()
    assert (b_e[0] == 0.0).all()
    assert (b_e[1] == 1.0).all()
    assert (b_e[2] == 2.0).all()
    assert (b_e[0] == 0.0).all()


def test_set_get_item():
    """
    Test setting of items.
    """
    t = torch.ones((2, 2))
    t_p = PackedTensor(t, 4, [2, 3])

    # Test setting channel.
    t_p[:, 0] = 4.0
    t_e = t_p.expand()
    assert t_e[0, 0] == 0.0
    assert t_e[2, 0] == 4.0
    assert t_e[2, 1] == 1.0

    # Test getting data from tensor.
    assert (t_p[:, 0] == 4.0).all()
    assert (t_p[..., 0] == 4.0).all()


def test_expand():
    """
    Test expansion of packed tensor.
    """
    indices = [0, 3]
    t = fill_tensor(torch.ones((2, 2)), indices)
    t_p = PackedTensor(t, 4, indices)
    t_e = t_p.expand()
    assert not isinstance(t_e, PackedTensor)
    assert t_e.shape[0] == 4
    assert (t_e[0] == 0.0).all()
    assert (t_e[3] == 3.0).all()

    # Empty tensor
    t = torch.ones((2, 2))
    t_p = PackedTensor(t, 4, [])
    t_e = t_p.expand()
    assert not isinstance(t_e, PackedTensor)
    assert t_e.shape[0] == 4
    assert (t_e == 0.0).all()


def test_intersection():
    """
    Test intersection of packed tensors.
    """
    indices = [0, 2]
    u = fill_tensor(torch.ones((2, 2)), indices)
    u_p = PackedTensor(u, 4, indices)

    indices = [2, 3]
    v = 2.0 * fill_tensor(torch.ones((2, 2)), indices)
    v_p = PackedTensor(v, 4, indices)
    u_i_p, v_i_p = u_p.intersection(v_p)
    assert u_i_p.batch_indices == [2]
    assert u_i_p.batch_size == 4

    u_i_e = u_i_p.expand()
    assert (u_i_e[2] == 2.0).all()

    v_i_e = v_i_p.expand()
    assert (v_i_e[2] == 2 * 2.0).all()

    u = torch.ones((2, 2))
    u_p = PackedTensor(u, 4, [0, 1])
    v = 2.0 * torch.ones((2, 2))
    v_p = PackedTensor(v, 4, [2, 3])
    u_i_p, v_i_p = u_p.intersection(v_p)
    assert u_i_p is None
    assert v_i_p is None

    for i in range(100):
        l = make_random_packed_tensor(100, 50)
        r = make_random_packed_tensor(100, 50)
        indices = sorted(list(set(l.batch_indices) & set(r.batch_indices)))
        l, r = l.intersection(r)
        for i, index in enumerate(indices):
            assert (l.tensor[i] == index).all()
            assert (r.tensor[i] == index).all()


def test_difference():
    """
    Test difference of packed tensors.
    """
    u = torch.zeros((2, 2))
    u[1] = 2.0
    u_p = PackedTensor(u, 4, [1, 3])
    v = 1.0 * torch.ones((2, 2))
    v_p = PackedTensor(v, 4, [0, 1])

    d_p = u_p.difference(v_p)
    assert d_p.batch_indices == [0, 3]
    assert d_p.batch_size == 4

    #d_e = d_p.expand()
    #assert (d_e[0] == 1.0).all()
    #assert (d_e[1] == 0.0).all()
    #assert (d_e[2] == 0.0).all()
    #assert (d_e[3] == 2.0).all()

    for i in range(10):
        l = make_random_packed_tensor(100, 50)
        r = make_random_packed_tensor(100, 50)
        indices = sorted(list(set(l.batch_indices) ^ set(r.batch_indices)))
        d = l.difference(r)
        for i, index in enumerate(indices):
            assert (d.tensor[i] == index).all()


def test_sum():
    """
    Test sum of packed tensors.
    """
    u = torch.ones((2, 2))
    u_p = PackedTensor(u, 4, [3])
    v = 2.0 * torch.ones((2, 2))
    v_p = PackedTensor(v, 4, [1, 2])

    s_p = u_p.sum(v_p)
    assert s_p.batch_indices == [1, 2, 3]
    assert s_p.batch_size == 4

    s_e = s_p.expand()
    assert (s_e[0] == 0.0).all()
    assert (s_e[1] == 2.0).all()
    assert (s_e[2] == 2.0).all()
    assert (s_e[3] == 1.0).all()

    s_p = v_p.sum(u_p)
    assert s_p.batch_indices == [1, 2, 3]
    assert s_p.batch_size == 4

    s_e = s_p.expand()
    assert (s_e[0] == 0.0).all()
    assert (s_e[1] == 2.0).all()
    assert (s_e[2] == 2.0).all()
    assert (s_e[3] == 1.0).all()

    for i in range(100):
        l = make_random_packed_tensor(100, 50)
        r = make_random_packed_tensor(100, 50)
        indices = sorted(list(set(l.batch_indices) | set(r.batch_indices)))
        s = l.sum(r)

        for i, index in enumerate(indices):
            assert (s.tensor[i] == index).all()


def test_apply_batch_norm():
    """
    Test application of batch norm layer, which requires the dim() member
    function.
    """
    t = make_random_packed_tensor(100, 50, (8, 16, 16))
    norm = nn.BatchNorm2d(8)
    norm.weight.data.fill_(0)
    norm.bias.data.fill_(1.0)
    y = norm(t)

    assert (y.tensor == 1.0).all()
