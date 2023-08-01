"""
quantnn.models.pytorch.upsampling
=================================

Upsampling factories.
"""
from torch import nn


class BilinearFactory:
    """
    A factory for producing bilinear upsampling layers.
    """
    def __call__(self, factor, channels_in=None, channels_out=None):
        if channels_in is not None:
            return nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=1),
                nn.UpsamplingBilinear2d(scale_factor=factor)
            )
        return nn.UpsamplingBilinear2d(scale_factor=factor)


class UpsampleFactory:
    """
    A factory for torch's generic upsampling layer.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, factor, channels_in=None, channels_out=None):
        if channels_in is not None:
            return nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=1),
                nn.Upsample(scale_factor=factor, **self.kwargs)
            )
        return nn.Upsample(scale_factor=factor, **self.kwargs)
