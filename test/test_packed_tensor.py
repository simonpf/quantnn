
import torch
from quantnn.packed_tensor import PackedTensor


def test_attributes():
    """
    Test attributes of packed tensor.
    """
    t = torch.ones((2, 2))
    t_p = PackedTensor(t, 4, [0, 1])

    assert t_p.batch_size == 4
    assert t_p.batch_indices == [0, 1]
    assert t_p.shape == (2, 2)

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
    t = torch.ones((2, 2))
    t_p = PackedTensor(t, 4, [0, 1])
    t_e = t_p.expand()
    assert not isinstance(t_e, PackedTensor)
    assert t_e.shape[0] == 4
    assert (t_e[2] == 0.0).all()
    assert (t_e[3] == 0.0).all()

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
    u = torch.ones((2, 2))
    u_p = PackedTensor(u, 4, [0, 1])
    v = 2.0 * torch.ones((2, 2))
    v_p = PackedTensor(v, 4, [1, 2])

    u_i_p, v_i_p = u_p.intersection(v_p)
    assert u_i_p.batch_indices == [1]
    assert u_i_p.batch_size == 4

    u_i_e = u_i_p.expand()
    assert (u_i_e[1] == 1.0).all()

    v_i_e = v_i_p.expand()
    assert (v_i_e[1] == 2.0).all()

    u = torch.ones((2, 2))
    u_p = PackedTensor(u, 4, [0, 1])
    v = 2.0 * torch.ones((2, 2))
    v_p = PackedTensor(v, 4, [2, 3])
    u_i_p, v_i_p = u_p.intersection(v_p)
    assert u_i_p is None
    assert v_i_p is None


def test_difference():
    """
    Test difference of packed tensors.
    """
    u = torch.ones((2, 2))
    u_p = PackedTensor(u, 4, [0, 1])
    v = 2.0 * torch.ones((2, 2))
    v_p = PackedTensor(v, 4, [1, 2])

    d_p = u_p.difference(v_p)
    assert d_p.batch_indices == [0, 2]
    assert d_p.batch_size == 4

    d_e = d_p.expand()
    assert (d_e[0] == 1.0).all()
    assert (d_e[1] == 0.0).all()
    assert (d_e[2] == 2.0).all()
    assert (d_e[3] == 0.0).all()


def test_sum():
    """
    Test sum of packed tensors.
    """
    u = torch.ones((2, 2))
    u_p = PackedTensor(u, 4, [0, 1])
    v = 2.0 * torch.ones((2, 2))
    v_p = PackedTensor(v, 4, [1, 2])

    s_p = u_p.sum(v_p)
    assert s_p.batch_indices == [0, 1, 2]
    assert s_p.batch_size == 4

    s_e = s_p.expand()
    assert (s_e[0] == 1.0).all()
    assert (s_e[1] == 1.0).all()
    assert (s_e[2] == 2.0).all()
    assert (s_e[3] == 0.0).all()
