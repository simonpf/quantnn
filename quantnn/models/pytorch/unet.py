"""
quantnn.models.pytorch.unet
===========================

This module provides an implementation of the UNet [unet]_
architecture.

.. [unet] O. Ronneberger, P. Fischer and T. Brox, "U-net: Convolutional networks
for biomedical image segmentation", Proc. Int. Conf. Med. Image Comput.
Comput.-Assist. Intervent. (MICCAI), pp. 234-241, 2015.
"""
import torch
import torch.nn as nn


def _conv2(channels_in, channels_out, kernel_size):
    """2D convolution with padding to keep image size constant. """
    return nn.Conv2d(
        channels_in,
        channels_out,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        padding_mode="reflect",
    )


class ConvolutionBlock(nn.Module):
    """
    A convolution block consisting of a pair of 2x2
    convolutions followed by a batch normalization layer and
    ReLU activaitons.
    """

    def __init__(self, channels_in, channels_out):
        """
        Create new convolution block.

        Args:
            channels_in: The number of input channels.
            channels_out: The number of output channels.
        """
        super().__init__()
        self.block = nn.Sequential(
            _conv2(channels_in, channels_out, 3),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            _conv2(channels_out, channels_out, 3),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Propagate input through layer."""
        return self.block(x)


class DownsamplingBlock(nn.Module):
    """
    UNet downsampling block consisting of 2x2 max-pooling followed
    by a convolution block.
    """

    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2), ConvolutionBlock(channels_in, channels_out)
        )

    def forward(self, x):
        """Propagate input through block."""
        return self.block(x)


class UpsamplingBlock(nn.Module):
    """
    UNet upsampling block consisting bilinear interpolation followed
    by a 1x1 convolution to decrease the channel dimensions and followed
    by a UNet convolution block.
    """

    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.upscaling = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.reduce = _conv2(channels_in, channels_in // 2, 3)
        self.conv = ConvolutionBlock(channels_in, channels_out)

    def forward(self, x, x_skip):
        """Propagate input through block."""
        x = self.reduce(self.upscaling(x))
        x = torch.cat([x, x_skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    PyTorch implementation of UNet, consisting of 4 downsampling
    blocks followed by 4 upsampling blocks and skip connection between
    down- and upsampling blocks of matching output and input size.

    The core of each down and upsampling block consists of two
    2D 3x3 convolution followed by batch norm and ReLU activation
    functions.
    """

    def __init__(self, n_inputs, n_outputs):
        """
        Args:
            n_input: The number of input channels.
            n_outputs: The number of output channels.
        """
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.in_block = ConvolutionBlock(n_inputs, 64)

        self.down_block_1 = DownsamplingBlock(64, 128)
        self.down_block_2 = DownsamplingBlock(128, 256)
        self.down_block_3 = DownsamplingBlock(256, 512)
        self.down_block_4 = DownsamplingBlock(512, 1024)

        self.up_block_1 = UpsamplingBlock(1024, 512)
        self.up_block_2 = UpsamplingBlock(512, 256)
        self.up_block_3 = UpsamplingBlock(256, 128)
        self.up_block_4 = UpsamplingBlock(128, n_outputs)

        self.out_block = _conv2(n_outputs, n_outputs, 1)

    def forward(self, x):
        """Propagate input through network."""

        d_64 = self.in_block(x)
        d_128 = self.down_block_1(d_64)
        d_256 = self.down_block_2(d_128)
        d_512 = self.down_block_3(d_256)
        d_1024 = self.down_block_4(d_512)

        u_512 = self.up_block_1(d_1024, d_512)
        u_256 = self.up_block_2(u_512, d_256)
        u_128 = self.up_block_3(u_256, d_128)
        u_out = self.up_block_4(u_128, d_64)

        return self.out_block(u_out)
