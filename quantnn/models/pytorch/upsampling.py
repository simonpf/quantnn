"""
quantnn.models.pytorch.upsampling
=================================

Upsampling factories.
"""
from torch import nn
from . import masked as nm


class BilinearFactory:
    """
    A factory for producing bilinear upsampling layers.
    """
    def __init__(self, masked=False):
        self.masked = masked

    def __call__(self, channels_in, channels_out, factor):
        if self.masked:
            mod = nm
        else:
            mod = nn

        if channels_in is not None and channels_out is not None:
            if channels_in != channels_out:
                return mod.Sequential(
                    mod.Conv2d(channels_in, channels_out, kernel_size=1),
                    mod.Upsample(scale_factor=factor, mode="bilinear"),
                )
        return mod.Upsample(scale_factor=factor, mode="bilinear")


class UpsampleFactory:
    """
    A factory for torch's generic upsampling layer.
    """

    def __init__(self, masked=False, **kwargs):
        self.kwargs = kwargs
        self.masked = masked

    def __call__(self, channels_in, channels_out, factor):
        if self.masked:
            mod = nm
        else:
            mod = nn

        if channels_in is not None and channels_out is not None:
            if channels_in != channels_out:
                return nn.Sequential(
                    mod.Conv2d(channels_in, channels_out, kernel_size=1),
                    mod.Upsample(scale_factor=factor, **self.kwargs),
                )
        return mod.Upsample(scale_factor=factor, **self.kwargs)


class UpConvolutionFactory:
    """
    Factory for up-convolution upsampling as used in the original
    U-Net architecture.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, channels_in, channels_out, factor):
        return nn.Sequential(
            nn.Upsample(scale_factor=factor, **self.kwargs),
            nn.Conv2d(channels_in, channels_out, kernel_size=factor, padding="same"),
        )
