"""
Tests for the quantnn.models.pytorch.base module.
"""
from torch import nn

from quantnn.models.pytorch.base import ParamCount


class ConvModule(nn.Conv2d, ParamCount):
    """
    Simple wrapper class that adds ParamCount mixin to
    a Conv2d module.
    """

    def __init__(
        self, channels_in: int, channels_out: int, kernel_size: int, bias: bool = False
    ):
        nn.Conv2d.__init__(
            self, channels_in, channels_out, kernel_size=kernel_size, bias=bias
        )


def test_n_params():
    """
    Ensure that n_params returns the right number of parameters for
    a 2D convolution layer.
    """
    conv = ConvModule(16, 16, 3, bias=True)
    assert conv.n_params == 16 * 16 * 3 * 3 + 16

    conv = ConvModule(16, 16, 3, bias=False)
    assert conv.n_params == 16 * 16 * 3 * 3
