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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import Input

from quantnn.models.keras.padding import SymmetricPadding


class ConvolutionBlock(layers.Layer):
    """
    A convolution block consisting of a pair of 3x3 convolutions followed by
    batch normalization and ReLU activations.
    """

    def __init__(self, channels_in, channels_out):
        """
        Create new convolution block.

        Args:
            channels_in: The number of input channels.
            channels_out: The number of output channels.
        """
        super().__init__()
        input_shape = (None, None, channels_in)
        self.block = keras.Sequential()
        self.block.add(SymmetricPadding(1))
        self.block.add(
            layers.Conv2D(channels_out, 3, padding="valid", input_shape=input_shape)
        )
        self.block.add(layers.BatchNormalization())
        self.block.add(layers.ReLU())
        self.block.add(SymmetricPadding(1))
        self.block.add(layers.Conv2D(channels_out, 3, padding="valid"))
        self.block.add(layers.BatchNormalization())
        self.block.add(layers.ReLU())

    def call(self, input):
        x = input
        return self.block(x)


class DownsamplingBlock(keras.Sequential):
    """
    A downsampling block consisting of a max pooling layer and a
    convolution block.
    """

    def __init__(self, channels_in, channels_out):
        """
        Create new convolution block.

        Args:
            channels_in: The number of input channels.
            channels_out: The number of output channels.
        """
        super().__init__()
        input_shape = (None, None, channels_in)
        self.add(layers.MaxPooling2D(strides=(2, 2)))
        self.add(ConvolutionBlock(channels_in, channels_out))


class UpsamplingBlock(layers.Layer):
    """
    An upsampling block which which uses bilinear interpolation
    to increase the input size. This is followed by a 1x1 convolution to
    reduce the number of channels, concatenation of the skip inputs
    from the corresponding downsampling layer and a convolution block.

    """

    def __init__(self, channels_in, channels_out):
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
            channels_in // 2, 1, padding="same", input_shape=input_shape
        )
        self.concat = layers.Concatenate()
        self.conv_block = ConvolutionBlock(channels_in, channels_out)

    def call(self, inputs):
        x, x_skip = inputs
        x_up = self.reduce(self.upsample(x))
        x = self.concat([x_up, x_skip])
        return self.conv_block(x)


class UNet(keras.Model):
    """
    Keras implementation of the UNet architecture, an input block followed
    by 4 encoder blocks and 4 decoder blocks.




    """

    def __init__(self, n_inputs, n_outputs):
        super().__init__()

        self.in_block = keras.Sequential(
            [
                SymmetricPadding(1),
                layers.Conv2D(64, 3, padding="valid", input_shape=(None, None, 128)),
                layers.BatchNormalization(),
                layers.ReLU(),
                SymmetricPadding(1),
                layers.Conv2D(64, 3, padding="valid"),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )

        self.down_block_1 = DownsamplingBlock(64, 128)
        self.down_block_2 = DownsamplingBlock(128, 256)
        self.down_block_3 = DownsamplingBlock(256, 512)
        self.down_block_4 = DownsamplingBlock(512, 1024)

        self.up_block_1 = UpsamplingBlock(1024, 512)
        self.up_block_2 = UpsamplingBlock(512, 256)
        self.up_block_3 = UpsamplingBlock(256, 128)
        self.up_block_4 = UpsamplingBlock(128, 64)

        self.out_block = layers.Conv2D(
            n_outputs, 1, padding="same", input_shape=(None, None, 64)
        )

    def call(self, inputs):

        d_64 = self.in_block(inputs)

        d_128 = self.down_block_1(d_64)
        d_256 = self.down_block_2(d_128)
        d_512 = self.down_block_3(d_256)
        d_1024 = self.down_block_4(d_512)

        u_512 = self.up_block_1([d_1024, d_512])
        u_256 = self.up_block_2([u_512, d_256])
        u_128 = self.up_block_3([u_256, d_128])
        u_64 = self.up_block_4([u_128, d_64])

        return self.out_block(u_64)
