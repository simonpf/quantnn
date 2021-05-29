import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SymmetricPadding(layers.Layer):
    def __init__(self, amount):
        super().__init__()
        self.paddings = tf.constant(
            [[0, 0], [amount, amount], [amount, amount], [0, 0]]
        )

    def call(self, input):
        return tf.pad(input, self.paddings, "SYMMETRIC")
