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
import keras
from keras import layers
from keras import activations
from keras import Input

class XceptionBlock(layers.Layer):
    """
    A convolution block consisting of a pair of 2x2
    convolutions followed by a batch normalization layer and
    ReLU activations.
    """
    def __init__(self,
                 channels_in,
                 channels_out,
                 downsample=False):
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
            self.block.add(layers.SeparableConv2D(channels_out, 3, padding="same",
                                                  strides=(2, 2), input_shape=input_shape))
        else:
            self.block.add(layers.SeparableConv2D(channels_out, 3, padding="same",
                                                  input_shape=input_shape))
        self.block.add(layers.BatchNormalization())
        self.block.add(layers.ReLU())
        self.block.add(layers.SeparableConv2D(channels_out, 3, padding="same",
                                              input_shape=input_shape))
        self.block.add(layers.BatchNormalization())
        self.block.add(layers.ReLU())

        if downsample:
            self.projection = layers.Conv2D(channels_out, 1, padding="same",
                                            input_shape=input_shape, strides=(2, 2))
        else:
            self.projection = layers.Conv2D(channels_out, 1, padding="same",
                                            input_shape=input_shape)

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
    def __init__(self,
                 channels_in,
                 channels_out,
                 n_blocks):
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
    def __init__(self,
                 channels_in,
                 channels_out,
                 n_blocks):
        """
        Create new convolution block.

        Args:
            channels_in: The number of input channels.
            channels_out: The number of output channels.
        """
        super().__init__()
        self.upsample = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        input_shape = (None, None, channels_in)
        self.reduce = layers.Conv2D(channels_in // 2, 1, padding="same", input_shape=input_shape)
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
    def __init__(self,
                 n_inputs,
                 n_outputs):
        super().__init__()

        self.in_block = keras.Sequential([
            layers.Conv2D(128, 5, input_shape=(None, None, n_inputs), padding="same"),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.down_block_1 = DownsamplingBlock(128, 256, 2)
        self.down_block_2 = DownsamplingBlock(256, 512, 2)
        self.down_block_3 = DownsamplingBlock(512, 1024, 2)
        self.down_block_4 = DownsamplingBlock(1024, 2048, 2)
        self.up_block_1 = UpsamplingBlock(2048, 1024, 2)
        self.up_block_2 = UpsamplingBlock(1024, 512, 2)
        self.up_block_3 = UpsamplingBlock(512, 256, 2)
        self.up_block_4 = UpsamplingBlock(256, 128, 2)

        self.concat = layers.Concatenate()
        self.out_block = keras.Sequential([
            layers.Conv2D(n_outputs, 1, padding="same", input_shape=(None, None, 128 + n_inputs)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(n_outputs, 1, padding="same")
            ])



    def call(self, inputs):

        d_32 = self.in_block(inputs)

        d_64 = self.down_block_1(d_32)
        d_128 = self.down_block_2(d_64)
        d_256 = self.down_block_3(d_128)
        d_512 = self.down_block_4(d_256)

        u_256 = self.up_block_1([d_512, d_256])
        u_128 = self.up_block_2([u_256, d_128])
        u_64 = self.up_block_3([u_128, d_64])
        u_32 = self.up_block_4([u_64, d_32])

        x_out = self.concat([u_32, inputs])
        return self.out_block(x_out)
