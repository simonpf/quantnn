"""
Tests for the quantnn.masked_tensor module.
"""
import numpy as np
import torch
from torch import nn

from quantnn.masked_tensor import MaskedTensor
import quantnn.models.pytorch.masked as nm

from quantnn.models.pytorch.encoders import SpatialEncoder
from quantnn.models.pytorch.decoders import SpatialDecoder
from quantnn.models.pytorch.blocks import ResNeXtBlockFactory
from quantnn.models.pytorch import upsampling, factories, normalization
from quantnn.models.pytorch.common import QuantileLoss
from quantnn.models.pytorch.fully_connected import MLP


def test_cat():
    """
    Test concatenation of masked tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mask = torch.rand(10, 10, 10) - 0.5 > 0
    masked_tensor = MaskedTensor(tensor, mask=mask)

    masked_tensor_2 = torch.cat([masked_tensor, masked_tensor], 1)
    assert masked_tensor_2.shape[1] == 20

    masked_tensor_2 = torch.cat([masked_tensor, tensor], 1)
    assert masked_tensor_2.shape[1] == 20

    masked_tensor_2 = torch.cat([tensor, masked_tensor, tensor], 1)
    assert masked_tensor_2.shape[1] == 30

    torch.cat([tensor, masked_tensor], 1, out=masked_tensor_2)
    assert masked_tensor_2.shape[1] == 20


def test_stack():
    """
    Test stacking of masked tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mask = torch.rand(10, 10, 10) - 0.5 > 0
    masked_tensor = MaskedTensor(tensor, mask=mask)

    masked_tensor_2 = torch.stack([masked_tensor, masked_tensor], 1)
    assert masked_tensor_2.shape[1] == 2

    masked_tensor_2 = torch.stack([masked_tensor, tensor], 1)
    assert masked_tensor_2.shape[1] == 2

    masked_tensor_2 = torch.stack([tensor, masked_tensor, tensor], 1)
    assert masked_tensor_2.shape[1] == 3

    torch.stack([tensor, masked_tensor], 1, out=masked_tensor_2)
    assert masked_tensor_2.shape[1] == 2


def test_add():
    """
    Test addition of masked tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mask_1 = torch.rand(10, 10, 10) - 0.5 > 0
    mask_2 = torch.rand(10, 10, 10) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)
    masked_tensor_2 = MaskedTensor(tensor, mask=mask_2)

    masked_tensor_2 = masked_tensor_1 + masked_tensor_2
    assert (masked_tensor_2.mask == (torch.logical_or(mask_1, mask_2))).all()
    assert torch.isclose(masked_tensor_2, 2.0 * tensor).all()


def test_mul():
    """
    Test multiplication of masked tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mask_1 = torch.rand(10, 10, 10) - 0.5 > 0
    mask_2 = torch.rand(10, 10, 10) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)
    masked_tensor_2 = MaskedTensor(tensor, mask=mask_2)

    masked_tensor_2 = masked_tensor_1 * masked_tensor_2
    assert (masked_tensor_2.mask == (torch.logical_or(mask_1, mask_2))).all()
    assert torch.isclose(masked_tensor_2, tensor**2).all()


def test_permute():
    """
    Test permutation of masked tensors.
    """
    tensor = torch.rand(1, 2, 3)
    mask_1 = torch.rand(1, 2, 3) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)
    masked_tensor_2 = torch.permute(masked_tensor_1, (2, 1, 0))
    assert masked_tensor_2.shape == (3, 2, 1)
    assert masked_tensor_2.mask.shape == (3, 2, 1)


def test_reshape():
    """
    Test reshaping of masked tensors.
    """
    tensor = torch.rand(1, 2, 3)
    mask_1 = torch.rand(1, 2, 3) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)
    masked_tensor_2 = torch.reshape(masked_tensor_1, (3, 2, 1))
    assert masked_tensor_2.shape == (3, 2, 1)
    assert masked_tensor_2.mask.shape == (3, 2, 1)


def test_view():
    """
    Test view applied to masked tensors.
    """
    tensor = torch.rand(1, 2, 3)
    mask_1 = torch.rand(1, 2, 3) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)
    masked_tensor_2 = masked_tensor_1.view((3, 2, 1))
    assert masked_tensor_2.shape == (3, 2, 1)
    assert masked_tensor_2.mask.shape == (3, 2, 1)


def test_squeeze():
    """
    Test squeezing of masked tensors.
    """
    tensor = torch.rand(1, 2, 3)
    mask_1 = torch.rand(1, 2, 3) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)
    masked_tensor_2 = masked_tensor_1.squeeze()
    assert masked_tensor_2.shape == (2, 3)

    masked_tensor_2 = torch.squeeze(masked_tensor_1)
    assert masked_tensor_2.shape == (2, 3)


def test_unsqueeze():
    """
    Test unsqueezing of masked tensors.
    """
    tensor = torch.rand(1, 2, 3)
    mask_1 = torch.rand(1, 2, 3) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)
    masked_tensor_2 = masked_tensor_1.unsqueeze(0)
    assert masked_tensor_2.shape == (1, 1, 2, 3)

    masked_tensor_2 = torch.unsqueeze(masked_tensor_1, 0)
    assert masked_tensor_2.shape == (1, 1, 2, 3)


def test_sum():
    """
    Test summing of tensor elements.
    """
    tensor = torch.rand(1, 2, 3)
    mask_1 = torch.rand(1, 2, 3) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)

    masked_sum = masked_tensor_1.sum()
    sum_ref = tensor[~mask_1].sum()
    assert torch.isclose(sum_ref, masked_sum)

    masked_sum = torch.sum(masked_tensor_1)
    sum_ref = torch.sum(tensor[~mask_1])
    assert torch.isclose(sum_ref, masked_sum)


def test_mean():
    """
    Test calculating the mean of a masked tensor.
    """
    tensor = torch.rand(1, 2, 3)
    mask_1 = torch.rand(1, 2, 3) - 0.5 > 0
    masked_tensor_1 = MaskedTensor(tensor, mask=mask_1)

    masked_mean = masked_tensor_1.mean()
    mean_ref = tensor[~mask_1].mean()
    assert torch.isclose(mean_ref, masked_mean)

    masked_mean = torch.mean(masked_tensor_1)
    mean_ref = torch.mean(tensor[~mask_1])
    assert torch.isclose(mean_ref, masked_mean)


def test_tensor_ops():
    """
    Test basic operations with tensors.
    """
    tensor = torch.rand(10, 10, 10)
    mask = torch.rand(10, 10, 10) - 0.5 > 0
    masked_tensor = MaskedTensor(tensor, mask=mask)
    assert hasattr(masked_tensor, "mask")

    masked_tensor_2 = MaskedTensor(masked_tensor)
    assert hasattr(masked_tensor_2, "mask")
    assert (masked_tensor.mask == masked_tensor_2.mask).all()

    tensor_2 = torch.rand(10, 10, 10)
    mask_2 = torch.rand(10, 10, 10) - 0.5 > 0
    masked_tensor_3 = MaskedTensor(tensor_2, mask=mask_2)

    masked_tensor_4 = masked_tensor_2 + masked_tensor_3
    mask = masked_tensor_4.mask
    assert torch.isfinite(masked_tensor_4[~mask]).all()
    assert (masked_tensor_4[~mask] == (tensor + tensor_2)[~mask]).all()

    masked_tensor_5 = masked_tensor_2 * masked_tensor_3
    mask = masked_tensor_5.mask
    assert (masked_tensor_5[~mask] == (tensor * tensor_2)[~mask]).all()

    masked_tensor_5 = masked_tensor[:5, :5]
    assert masked_tensor_5.mask.shape == (5, 5, 10)

    masked_tensor_6 = torch.permute(masked_tensor_5, (2, 0, 1))
    assert masked_tensor_6.shape == (10, 5, 5)
    assert masked_tensor_6.mask.shape == (10, 5, 5)

    masked_tensor_7 = masked_tensor_5.reshape((50, 5))
    assert masked_tensor_7.shape == (50, 5)
    assert masked_tensor_7.mask.shape == (50, 5)

    masked_tensor_7 += 1.0

    masked_tensor_8 = torch.stack([masked_tensor_7, masked_tensor_7])
    assert masked_tensor_8.shape == (2, 50, 5)
    assert masked_tensor_8.mask.shape == (2, 50, 5)


def test_conv2d():
    """
    Test masked 2D convolution.
    """
    tensor = torch.rand(1, 10, 10, 10)
    mask = torch.rand(1, 10, 10, 10) - 0.5 > 0
    tensor[mask] = np.nan
    t_m = MaskedTensor(tensor, mask=mask)

    conv2d = nn.Conv2d(10, 10, 3)
    y = conv2d(tensor)

    conv2d_m = nm.Conv2d(10, 10, 3)
    y_m = conv2d_m(t_m)

    assert torch.isnan(y).any()
    assert torch.isfinite(y_m).all()

    y.sum().backward()
    assert torch.isnan(conv2d.weight.grad).any()

    y_m.sum().backward()
    assert torch.isfinite(conv2d_m.weight.grad).all()


def test_linear():
    """
    Test masked linear layer.
    """
    tensor = torch.rand(10, 10)
    mask = torch.rand(10, 10) - 0.5 > 0
    tensor[mask] = np.nan
    t_m = MaskedTensor(tensor, mask=mask)

    linear = nn.Linear(10, 10)
    y = linear(tensor)

    linear_m = nm.Linear(10, 10)
    y_m = linear_m(t_m)

    assert torch.isnan(y).any()
    assert torch.isfinite(y_m).all()

    y.sum().backward()
    assert torch.isnan(linear.weight.grad).any()

    y_m.sum().backward()
    assert torch.isfinite(linear_m.weight.grad).all()
    assert torch.isfinite(linear_m.sep_bias.grad).all()

    tensor = torch.rand(10, 10)
    mask = torch.rand(10, 10) > 1.0
    tensor[mask] = np.nan
    t_m = MaskedTensor(tensor, mask=mask)
    linear_m.weight.data[:] = linear.weight.data
    linear_m.sep_bias.data[:] = linear.bias.data

    y = linear(tensor)
    y_m = linear_m(t_m)

    assert torch.isclose(y, y_m, atol=1e-3).all()


def test_batch_norm_1d():
    """
    Test masked 2D batch norm.
    """
    tensor = torch.arange(10)[..., None] + torch.rand(10, 10)
    mask = torch.rand(10, 10) - 0.5 > 0
    tensor[mask] = np.nan
    t_m = MaskedTensor(tensor, mask=mask)

    bn = nn.BatchNorm1d(10)
    y = bn(tensor)

    bn_m = nm.BatchNorm1d(10)
    y_m = bn_m(t_m)
    assert isinstance(y_m, MaskedTensor)

    assert torch.isnan(y).all()
    assert torch.isfinite(y_m).all()

    y.sum().backward()
    assert torch.isnan(bn.weight.grad).any()

    y_m.sum().backward()
    assert torch.isfinite(bn_m.weight.grad).all()


def test_batch_norm_2d():
    """
    Test masked 2D batch norm.
    """
    tensor = torch.arange(2)[None, ..., None, None] + torch.rand(10, 2, 10, 10)
    mask = torch.rand(10, 2, 10, 10) - 0.5 > 0
    tensor[mask] = np.nan
    t_m = MaskedTensor(tensor, mask=mask)

    bn = nn.BatchNorm2d(2)
    y = bn(tensor)

    bn_m = nm.BatchNorm2d(2)
    y_m = bn_m(t_m)
    assert isinstance(y_m, MaskedTensor)

    assert torch.isnan(y).all()
    assert torch.isfinite(y_m).all()

    y.sum().backward()
    assert torch.isnan(bn.weight.grad).any()

    y_m.sum().backward()
    assert torch.isfinite(bn_m.weight.grad).all()

    tensor = torch.arange(2)[None, ..., None, None] + torch.rand(8, 2, 64, 64)
    mask = torch.rand(8, 2, 64, 64) > 1.0
    t_m = MaskedTensor(tensor, mask=mask)
    y = bn(tensor)
    y_m = bn_m(t_m)
    assert torch.isclose(y, y_m, atol=1e-3).all()


def test_layer_norm():
    """
    Test masked 2D layer norm.
    """
    tensor = torch.arange(10)[None, ..., None, None] + torch.rand(10, 10, 10, 10)
    mask = torch.rand(10, 10, 10, 10) - 0.5 > 0
    tensor[mask] = np.nan
    t_m = MaskedTensor(tensor, mask=mask)

    ln = normalization.LayerNormFirst(10)
    y = ln(tensor)

    ln_m = nm.LayerNormFirst(10)
    y_m = ln_m(t_m)
    assert isinstance(y_m, MaskedTensor)

    assert torch.isnan(y).any()
    assert torch.isfinite(y_m).any()

    y.sum().backward()
    assert torch.isnan(ln.scaling.grad).any()

    assert torch.isfinite(y_m[~y_m.mask]).all()
    y_m_s = y_m[~y_m.mask].sum()
    y_m_s.backward()
    assert torch.isfinite(ln_m.scaling.grad).all()


def test_avg_pool_2d():
    """
    Test masked average pooling.
    """
    tensor = torch.arange(10)[None, ..., None, None] + torch.rand(10, 10, 10, 10)
    tensor[..., ::2] = np.nan
    mask = torch.isnan(tensor)
    t_m = MaskedTensor(tensor, mask=mask)

    mp = nn.AvgPool2d(kernel_size=2, stride=2)
    y = mp(tensor)

    mp_m = nm.AvgPool2d(kernel_size=2, stride=2)
    y_m = mp_m(t_m)

    assert isinstance(y_m, MaskedTensor)
    assert y_m.shape == (10, 10, 5, 5)
    assert y_m.mask.shape == (10, 10, 5, 5)
    assert torch.isfinite(y_m[y_m.mask]).all()
    assert torch.isfinite(y_m).any()

    tensor = torch.arange(10)[None, ..., None, None] + torch.rand(10, 10, 10, 10)
    mask = torch.zeros(tensor.shape, dtype=bool)
    t_m = MaskedTensor(tensor, mask=mask)
    y = mp(tensor)
    y_m = mp_m(t_m)
    assert (y_m == y).all()


def test_max_pool_2d():
    """
    Test masked 2D batch norm.
    """
    tensor = torch.arange(10)[None, ..., None, None] + torch.rand(10, 10, 10, 10)
    tensor[..., ::2] = np.nan
    mask = torch.isnan(tensor)
    t_m = MaskedTensor(tensor, mask=mask)

    mp = nn.MaxPool2d(kernel_size=2, stride=2)
    y = mp(tensor)

    mp_m = nm.MaxPool2d(kernel_size=2, stride=2)
    y_m = mp_m(t_m)
    assert isinstance(y_m, MaskedTensor)

    assert y_m.shape == (10, 10, 5, 5)
    assert y_m.mask.shape == (10, 10, 5, 5)

    assert torch.isnan(y).all()
    assert torch.isfinite(y_m).all()


def test_upsample():
    """
    Test upsampling layers.
    """
    tensor = torch.arange(10)[None, ..., None, None] + torch.rand(10, 10, 10, 10)
    tensor[..., ::2] = np.nan
    mask = torch.isnan(tensor)
    t_m = MaskedTensor(tensor, mask=mask)

    up = nn.Upsample(scale_factor=2, mode="bilinear")
    y = up(tensor)

    mp_m = nm.Upsample(scale_factor=2, mode="bilinear")
    y_m = mp_m(t_m)
    assert isinstance(y_m, MaskedTensor)

    assert torch.isnan(y).any()
    assert torch.isfinite(y_m[~y_m.mask]).all()


def test_quantile_loss():
    x_1 = torch.rand((1, 8, 128, 128))
    mask = (x_1 - 0.5) < 0.0
    x_1[mask] = np.nan
    x_1_m = MaskedTensor(x_1, mask=mask)

    x_2 = torch.rand((1, 1, 128, 128))
    mask = (x_2 - 0.5) < 0.0
    x_2[mask] = np.nan
    x_2_m = MaskedTensor(x_2, mask=mask)

    loss = QuantileLoss(np.linspace(0, 1, 10)[1:-1], mask=-100)
    ql = loss(x_1, x_2)
    ql_m = loss(x_1_m, x_2_m)

    assert torch.isnan(ql).all()
    assert torch.isfinite(ql_m).all()


def test_quantile_resnext():
    block_factory = ResNeXtBlockFactory(cardinality=4, masked=True)
    downsampler_factory = factories.MaxPooling(masked=True)
    upsampler_factory = upsampling.UpsampleFactory(masked=True)

    class SimpleResnext(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = SpatialEncoder(
                channels=[8, 16, 32],
                downsampling_factors=[2, 2],
                stages=[1, 1, 1],
                block_factory=block_factory,
                downsampler_factory=downsampler_factory,
            )
            self.dec = SpatialDecoder(
                channels=[8, 16, 32][::-1],
                upsampling_factors=[2, 2],
                stages=[1, 1],
                skip_connections=self.enc.skip_connections,
                block_factory=block_factory,
                upsampler_factory=upsampler_factory,
            )
            self.head = MLP(8, 16, 16, 3, masked=True)

        def forward(self, x):
            enc = self.enc(x, return_skips=True)
            return self.head(self.dec(self.enc(x, return_skips=True)))

    x = torch.rand((2, 8, 128, 128))
    mask = (x - 0.5) < 0.0
    x[mask] = np.nan
    x_m = MaskedTensor(x, mask=mask)

    resnext = SimpleResnext()

    y = resnext(x)
    assert torch.isnan(y).all()

    y_m = resnext(x_m)

    y_true = torch.rand((2, 128, 128))

    ql = QuantileLoss(np.arange(0, 1, 10)[1:-1])
    assert torch.isfinite(y_m).all()


def test_compress():
    x = torch.rand((4, 8, 128, 128))
    x[0] = np.nan
    x[2] = np.nan
    mask = torch.isnan(x)
    x_m = MaskedTensor(x, mask=mask)

    x_c = x_m.compress()
    assert x_c.shape == (2, 8, 128, 128)

    x_m_2 = x_c.decompress()

    assert (x_m.mask == x_m_2.mask).all()
    assert (x_m[~x_m.mask] == x_m_2[~x_m_2.mask]).all()
    assert x_m_2.shape == (4, 8, 128, 128)
