from enum import Enum, auto
from TensorNAS.Core.Layer import Layer


class Args(Enum):

    NONE = auto()


class Layer(Layer):
    def _gen_args(self, input_shape, args):
        return {}

    def get_output_shape(self):
        return self.inputshape.get()

    def get_keras_layers(self, input_tensor):
        import tensorflow as tf

        return tf.keras.layers.BatchNormalization()(input_tensor)
