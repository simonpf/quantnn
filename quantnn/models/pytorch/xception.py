"""
===============================
quantnn.models.pytorch.xception
===============================

PyTorch implementation of models using the Xception architecture.
"""
import torch
from torch import nn

class SymmetricPadding(nn.Module):
    def __init__(self, amount):
        super().__init__()
        if isinstance(amount, int):
            self.amount = [amount] * 4
        else:
            self.amount = amount

    def forward(self, x):
        return nn.functional.pad(x, self.amount, "replicate")

def sconv3x3(channels_in, channels_out):
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_in, kernel_size=3,
                  groups=channels_in, padding=1,
                  padding_mode="replicate"),
        nn.Conv2d(channels_in, channels_out, kernel_size=1)
    )

class XceptionBlock(nn.Module):
    def __init__(self, channels_in, channels_out, downsample=False):
        super().__init__()
        if downsample:
            self.block_1 = nn.Sequential(
                sconv3x3(channels_in, channels_out),
                nn.BatchNorm2d(channels_out),
                SymmetricPadding(1),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.ReLU()
                )
        else:
            self.block_1 = nn.Sequential(
                sconv3x3(channels_in, channels_out),
                nn.BatchNorm2d(channels_out),
                nn.ReLU()
                )

        self.block_2 = nn.Sequential(
            sconv3x3(channels_out, channels_out),
            nn.BatchNorm2d(channels_out),
            nn.ReLU()
            )

        if channels_in != channels_out or downsample:
            if downsample:
                self.projection = nn.Conv2d(channels_in, channels_out, 1,
                                            stride=2)
            else:
                self.projection = nn.Conv2d(channels_in, channels_out, 1)
        else:
            self.projection = None

    def forward(self, x):
        if self.projection is None:
            x_proj = x
        else:
            x_proj = self.projection(x)
        y = self.block_2(self.block_1(x))
        return x_proj + y

class DownsamplingBlock(nn.Sequential):
    def __init__(self, n_channels, n_blocks):
        blocks = [XceptionBlock(n_channels, n_channels, downsample=True)]
        for i in range(n_blocks):
            blocks.append(XceptionBlock(n_channels, n_channels))
        super().__init__(*blocks)

class UpsamplingBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.upsample = nn.Upsample(mode="bilinear", scale_factor=2)
        self.block = nn.Sequential(
            sconv3x3(n_channels * 2, n_channels),
            nn.BatchNorm2d(n_channels),
            nn.ReLU()
        )

    def forward(self, x, x_skip):
        x_up = self.upsample(x)
        x_merged = torch.cat([x_up, x_skip], 1)
        return self.block(x_merged)


class XceptionFpn(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_features=128,
                 blocks=2):

        super().__init__()

        if isinstance(blocks, int):
            blocks = [blocks] * 5

        self.in_block = nn.Conv2d(n_inputs, n_features, 1)

        self.down_block_2 = DownsamplingBlock(n_features, blocks[0])
        self.down_block_4 = DownsamplingBlock(n_features, blocks[1])
        self.down_block_8 = DownsamplingBlock(n_features, blocks[2])
        self.down_block_16 = DownsamplingBlock(n_features, blocks[3])
        self.down_block_32 = DownsamplingBlock(n_features, blocks[4])

        self.up_block_16 = UpsamplingBlock(n_features)
        self.up_block_8 = UpsamplingBlock(n_features)
        self.up_block_4 = UpsamplingBlock(n_features)
        self.up_block_2 = UpsamplingBlock(n_features)
        self.up_block = UpsamplingBlock(n_features)

        self.head = nn.Sequential(
            nn.Conv2d(2 * n_features, n_features, 1),
            nn.BatchNorm2d(n_features),
            nn.ReLU(),
            nn.Conv2d(n_features, n_features, 1),
            nn.BatchNorm2d(n_features),
            nn.ReLU(),
            nn.Conv2d(n_features, n_outputs, 1)
        )


    def forward(self, x):
        x_in = self.in_block(x)
        x_2 = self.down_block_2(x_in)
        x_4 = self.down_block_4(x_2)
        x_8 = self.down_block_8(x_4)
        x_16 = self.down_block_16(x_8)
        x_32 = self.down_block_32(x_16)

        x_16_u = self.up_block_16(x_32, x_16)
        x_8_u = self.up_block_8(x_16_u, x_8)
        x_4_u = self.up_block_4(x_8_u, x_4)
        x_2_u = self.up_block_2(x_4_u, x_2)
        x_u = self.up_block(x_2_u, x_in)

        return self.head(torch.cat([x_in, x_u], 1))


model = XceptionFpn(1, 1)


