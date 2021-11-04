from enum import Enum, auto

from TensorNAS.Core.Layer import Layer
from TensorNAS.Tools.Util import dimension_mag


class Args(Enum):
    NONE = auto()


class Layer(Layer):
    def _gen_args(self, input_shape, args):
        return {}

    def get_output_shape(self):
        return (1, dimension_mag(self.inputshape.get()))

    def get_keras_layers(self, input_tensor):
        import tensorflow as tf

        return tf.keras.layers.Flatten(input_shape=self.inputshape.get())(input_tensor)
