"""
quantnn.models.pytorch.resnet
=============================

This module provides an implementation of a fully-covolutional
[ResNet]_-based decoder-encoder architecture.


.. [ResNet] Deep Residual Learning for Image Recognition
"""
import torch
import torch.nn as nn


def _conv2(channels_in, channels_out, kernel_size):
    """
    Convolution with reflective padding to keep image size constant.
    """
    return nn.Conv2d(
        channels_in,
        channels_out,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        padding_mode="reflect",
    )


def _conv2_down(channels_in, channels_out, kernel_size):
    """
    Convolution combined with downsampling and reflective padding to
    decrease input size by a factor of 2.
    """
    return nn.Conv2d(
        channels_in,
        channels_out,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        stride=2,
        padding_mode="reflect",
    )


class ResidualBlock(nn.Module):
    """
    A residual block consists of either two or three convolution operations
    followed by batch norm and relu activation together with an identity
    mapping connecting the input and the activation feeding into the
    last ReLU layer.
    """

    def __init__(self, channels_in, channels_out, bottleneck=None, downsample=False):
        """
        Create new convolution block.

        Args:
            channels_in: The number of input channels.
            channels_out: The number of output channels.
            bottleneck: Whether to apply a bottle neck to reduce the
                number of parameters.
            downsample: If true the image dimensions are reduced by
                a factor of two.
        """
        super().__init__()
        self.downsample = downsample
        if bottleneck is None:
            self.block = nn.Sequential(
                _conv2(channels_in, channels_out, 3),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(inplace=True),
                (
                    _conv2_down(channels_out, channels_out, 3)
                    if downsample
                    else _conv2(channels_out, channels_out, 3)
                ),
                nn.BatchNorm2d(channels_out),
            )
        else:
            self.block = nn.Sequential(
                _conv2(channels_in, bottleneck, 1),
                nn.BatchNorm2d(bottleneck),
                nn.ReLU(inplace=True),
                _conv2(bottleneck, bottleneck, 3),
                nn.BatchNorm2d(bottleneck),
                nn.ReLU(inplace=True),
                (
                    _conv2_down(bottleneck, channels_out, 1)
                    if downsample
                    else _conv2(bottleneck, channels_out, 1)
                ),
                nn.BatchNorm2d(channels_out),
            )
        self.activation = nn.ReLU(inplace=True)

        self.projection = None
        if channels_in != channels_out:
            if downsample:
                self.projection = _conv2_down(channels_in, channels_out, 1)
            else:
                self.projection = _conv2(channels_in, channels_out, 1)

    def forward(self, x):
        """
        Propagate input through block.
        """
        y = self.block(x)
        if self.projection:
            x = self.projection(x)
        return self.activation(y + x)


class DownSamplingBlock(nn.Module):
    """
    UNet downsampling block consisting of strided convolution followed
    by given number of residual blocks.
    """

    def __init__(self, channels_in, channels_out, n_blocks, bottleneck=None):
        super().__init__()
        modules = [
            ResidualBlock(
                channels_in, channels_out, bottleneck=bottleneck, downsample=True
            )
        ] * (n_blocks - 1)
        modules += [
            ResidualBlock(channels_out, channels_out, bottleneck=bottleneck)
        ] * (n_blocks - 1)
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        """Propagate input through block."""
        return self.block(x)


class UpSamplingBlock(nn.Module):
    """
    ResNet upsampling block consisting of linear interpolation
    followed by given number of residual blocks.
    """

    def __init__(
        self, channels_in, channels_skip, channels_out, n_blocks, bottleneck=None
    ):
        super().__init__()
        self.upscaling = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

        modules = [
            ResidualBlock(
                channels_in + channels_skip, channels_out, bottleneck=bottleneck
            )
        ]
        modules += [
            ResidualBlock(channels_out, channels_out, bottleneck=bottleneck)
        ] * (n_blocks - 1)
        self.block = nn.Sequential(*modules)

    def forward(self, x, x_skip):
        """Propagate input through block."""
        x = self.upscaling(x)
        x = torch.cat([x, x_skip], dim=1)
        return self.block(x)


class ResNet(nn.Module):
    """
    Decoder-encoder network using residual blocks.

    The ResNet class implements a fully-convolutional decoder-encoder
    network for point-to-point regression tasks.

    The network consists of 5 downsampling blocks followed by the
    same number of upsampling blocks. The first downsampling block
    consists of a 7x7 convolution with stride two followed by batch
    norm and ReLU activation. All following block are residual blocks
    with bottlenecks.
    """

    def __init__(self, n_inputs, n_outputs, blocks=2):

        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.in_block = nn.Sequential(
            _conv2_down(n_inputs, 128, 7), nn.BatchNorm2d(128), nn.ReLU()
        )

        if type(blocks) is int:
            blocks = 4 * [blocks]

        self.down_block_1 = DownSamplingBlock(128, 256, blocks[0], bottleneck=256)
        self.down_block_2 = DownSamplingBlock(256, 512, blocks[1], bottleneck=256)
        self.down_block_3 = DownSamplingBlock(512, 1024, blocks[2], bottleneck=256)
        self.down_block_4 = DownSamplingBlock(1024, 2048, blocks[3], bottleneck=512)

        self.up_block_1 = UpSamplingBlock(2048, 1024, 1024, blocks[3], bottleneck=256)
        self.up_block_2 = UpSamplingBlock(1024, 512, 512, blocks[2], bottleneck=256)
        self.up_block_3 = UpSamplingBlock(512, 256, 256, blocks[1], bottleneck=256)
        self.up_block_4 = UpSamplingBlock(256, 128, n_outputs, blocks[0])
        self.up_block_5 = UpSamplingBlock(n_outputs, n_inputs, n_outputs, blocks[0])

        self.out_block = nn.Sequential(
            _conv2(n_outputs, n_outputs, 1),
            nn.BatchNorm2d(n_outputs),
            nn.ReLU(),
            _conv2(n_outputs, n_outputs, 1),
        )

    def forward(self, x):
        """Propagate input through resnet."""

        d_0 = self.in_block(x)
        d_1 = self.down_block_1(d_0)
        d_2 = self.down_block_2(d_1)
        d_3 = self.down_block_3(d_2)
        d_4 = self.down_block_4(d_3)

        u_4 = self.up_block_1(d_4, d_3)
        u_3 = self.up_block_2(u_4, d_2)
        u_2 = self.up_block_3(u_3, d_1)
        u_1 = self.up_block_4(u_2, d_0)
        u_0 = self.up_block_5(u_1, x)

        return self.out_block(u_0)
