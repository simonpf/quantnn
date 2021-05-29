"""
=========================
quantnn.models.keras.unet
=========================

This module provides an implementation of the UNet [unet]_
architecture.

.. [unet] O. Ronneberger, P. Fischer and T. Brox, "U-net: Convolutional networks
for biomedical image segmentation", Proc. Int. Conf. Med. Image Comput.
Comput.-Assist. Intervent. (MICCAI), pp. 234-241, 2015.
"""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import Input

from quantnn.models.keras.padding import SymmetricPadding


class XceptionBlock(layers.Layer):
    """
    A convolution block consisting of a pair of 2x2
    convolutions followed by a batch normalization layer and
    ReLU activations.
    """

    def __init__(self, channels_in, channels_out, downsample=False):
        """
        Create new convolution block.

        Args:
            channels_in: The number of input channels.
            channels_out: The number of output channels.
        """
        super().__init__()
        input_shape = (None, None, channels_in)

        self.block = keras.Sequential()
        if downsample:
            self.block.add(SymmetricPadding(1))
            self.block.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
            self.block.add(SymmetricPadding(1))
            self.block.add(
                layers.SeparableConv2D(
                    channels_out, 3, padding="valid", input_shape=input_shape
                )
            )
        else:
            self.block.add(SymmetricPadding(1))
            self.block.add(
                layers.SeparableConv2D(
                    channels_out, 3, padding="valid", input_shape=input_shape
                )
            )
        self.block.add(layers.BatchNormalization())
        self.block.add(layers.ReLU())
        self.block.add(SymmetricPadding(1))
        self.block.add(
            layers.SeparableConv2D(
                channels_out, 3, padding="valid", input_shape=input_shape
            )
        )
        self.block.add(layers.BatchNormalization())
        self.block.add(layers.ReLU())

        if downsample:
            self.projection = layers.Conv2D(
                channels_out,
                1,
                padding="valid",
                input_shape=input_shape,
                strides=(2, 2),
            )
        else:
            self.projection = layers.Conv2D(
                channels_out, 1, padding="valid", input_shape=input_shape
            )

    def call(self, inputs):
        x = inputs
        x_proj = self.projection(x)
        y = self.block(x)
        return x_proj + y


class DownsamplingBlock(keras.Sequential):
    """
    A downsampling block consisting of a max pooling layer and a
    convolution block.
    """

    def __init__(self, channels_in, channels_out, n_blocks):
        """
        Create new convolution block.

        Args:
            channels_in: The number of input channels.
            channels_out: The number of output channels.
        """
        super().__init__()
        input_shape = (None, None, channels_in)
        self.add(XceptionBlock(channels_in, channels_out, downsample=True))
        for i in range(n_blocks):
            self.add(XceptionBlock(channels_out, channels_out))


class UpsamplingBlock(layers.Layer):
    """
    An upsampling block which which uses bilinear interpolation
    to increase the input size. This is followed by a 1x1 convolution to
    reduce the number of channels, concatenation of the skip inputs
    from the corresponding downsampling layer and a convolution block.

    """

    def __init__(self, channels_in, channels_out, n_blocks):
        """
        Create new convolution block.

        Args:
            channels_in: The number of input channels.
            channels_out: The number of output channels.
        """
        super().__init__()
        self.upsample = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        input_shape = (None, None, channels_in)
        self.reduce = layers.Conv2D(
            channels_in // 2, 1, padding="valid", input_shape=input_shape
        )
        self.concat = layers.Concatenate()

        self.blocks = keras.Sequential()
        self.blocks.add(XceptionBlock(channels_in, channels_out))
        for i in range(n_blocks - 1):
            self.blocks.add(XceptionBlock(channels_out, channels_out))

    def call(self, inputs):
        x, x_skip = inputs
        x_up = self.reduce(self.upsample(x))
        x = self.concat([x_up, x_skip])
        return self.blocks(x)


class XceptionNet(keras.Model):
    """
    Keras implementation of the UNet architecture, an input block followed
    by 4 encoder blocks and 4 decoder blocks.




    """

    def __init__(self, n_inputs, n_outputs, n_base_features=64):
        super().__init__()

        nf = n_base_features

        self.in_block = keras.Sequential(
            [
                SymmetricPadding(2),
                layers.Conv2D(
                    nf, 5, input_shape=(None, None, n_inputs), padding="valid"
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )

        self.down_block_1 = DownsamplingBlock(nf, 2 * nf, 2)
        self.down_block_2 = DownsamplingBlock(2 * nf, 4 * nf, 2)
        self.down_block_3 = DownsamplingBlock(4 * nf, 8 * nf, 2)
        self.down_block_4 = DownsamplingBlock(8 * nf, 16 * nf, 2)
        self.down_block_5 = DownsamplingBlock(16 * nf, 32 * nf, 2)

        self.up_block_1 = UpsamplingBlock(32 * nf, 16 * nf, 2)
        self.up_block_2 = UpsamplingBlock(16 * nf, 8 * nf, 2)
        self.up_block_3 = UpsamplingBlock(8 * nf, 4 * nf, 2)
        self.up_block_4 = UpsamplingBlock(4 * nf, 2 * nf, 2)
        self.up_block_5 = UpsamplingBlock(2 * nf, nf, 2)

        self.concat = layers.Concatenate()
        self.out_block = keras.Sequential(
            [
                layers.Conv2D(
                    n_outputs,
                    1,
                    padding="valid",
                    input_shape=(None, None, nf + n_inputs),
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(n_outputs, 1, padding="valid"),
            ]
        )

    def call(self, inputs):

        d_32 = self.in_block(inputs)

        d_64 = self.down_block_1(d_32)
        d_128 = self.down_block_2(d_64)
        d_256 = self.down_block_3(d_128)
        d_512 = self.down_block_4(d_256)
        d_1024 = self.down_block_5(d_512)

        u_512 = self.up_block_1([d_1024, d_512])
        u_256 = self.up_block_2([u_512, d_256])
        u_128 = self.up_block_3([u_256, d_128])
        u_64 = self.up_block_4([u_128, d_64])
        u_32 = self.up_block_5([u_64, d_32])

        x_out = self.concat([u_32, inputs])
        return self.out_block(x_out)


class FpnUpsamplingBlock(layers.Layer):
    """
    An upsampling block which which uses bilinear interpolation
    to increase the input size. This is followed by a 1x1 convolution to
    reduce the number of channels, concatenation of the skip inputs
    from the corresponding downsampling layer and a convolution block.

    """

    def __init__(self, channels, n):
        """
        Create new convolution block.

        Args:
            channels_in: The number of input channels.
            channels_out: The number of output channels.
        """
        super().__init__()
        self.upsampler = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")

        self.concat = layers.Concatenate()

        self.block = keras.Sequential()
        self.block.add(SymmetricPadding(1))
        self.block.add(
            layers.SeparableConv2D(channels, 3, input_shape=(None, None, 2 * channels))
        )
        self.block.add(layers.BatchNormalization())
        self.block.add(layers.ReLU())

    def call(self, x_coarse, x_fine):
        x_up = self.upsampler(x_coarse)
        return self.block(self.concat([x_up, x_fine]))


class XceptionFpn(keras.Model):
    def __init__(self, n_inputs, n_outputs, n_base_features=64, blocks=2):
        super().__init__()

        nf = n_base_features

        self.in_block = layers.Conv2D(nf, 1, input_shape=(None, None, n_inputs))

        if isinstance(blocks, int):
            blocks = [blocks] * 5

        self.down_block_2 = DownsamplingBlock(nf, nf, blocks[0])
        self.down_block_4 = DownsamplingBlock(nf, nf, blocks[1])
        self.down_block_8 = DownsamplingBlock(nf, nf, blocks[2])
        self.down_block_16 = DownsamplingBlock(nf, nf, blocks[3])
        self.down_block_32 = DownsamplingBlock(nf, nf, blocks[4])

        self.up_block_32 = FpnUpsamplingBlock(nf, 5)
        self.up_block_16 = FpnUpsamplingBlock(nf, 4)
        self.up_block_8 = FpnUpsamplingBlock(nf, 3)
        self.up_block_4 = FpnUpsamplingBlock(nf, 2)
        self.up_block_2 = FpnUpsamplingBlock(nf, 1)

        self.concat = layers.Concatenate()
        self.out_block = keras.Sequential(
            [
                layers.Conv2D(
                    nf, 1, padding="valid", input_shape=(None, None, nf + n_inputs)
                ),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(nf, 1, padding="valid", input_shape=(None, None, nf)),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(n_outputs, 1, padding="valid"),
            ]
        )

    def call(self, inputs):

        x_p = self.in_block(inputs)

        x_2 = self.down_block_2(x_p)
        x_4 = self.down_block_4(x_2)
        x_8 = self.down_block_8(x_4)
        x_16 = self.down_block_16(x_8)
        x_32 = self.down_block_32(x_16)

        x_16_u = self.up_block_32(x_32, x_16)
        x_8_u = self.up_block_16(x_16_u, x_8)
        x_4_u = self.up_block_8(x_8_u, x_4)
        x_2_u = self.up_block_4(x_4_u, x_2)
        x_u = self.up_block_2(x_2_u, x_p)

        return self.out_block(self.concat([x_u, inputs]))
