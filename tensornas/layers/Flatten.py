import tensorflow as tf
from enum import Enum, auto

from tensornas.core.networklayer import NetworkLayer
from tensornas.core.util import dimension_mag


class Args(Enum):
    NONE = auto()


class Layer(NetworkLayer):
    def _gen_args(cls, input_shape, args):
        return {}

    def get_output_shape(self):
        return (1, dimension_mag(self.inputshape.get()))

    def get_keras_layer(self):
        return tf.keras.layers.Flatten()
