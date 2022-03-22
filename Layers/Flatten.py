from TensorNAS.Core import EnumWithNone
from TensorNAS.Core.Layer import Layer
from enum import auto


class Args(EnumWithNone):

    NONE = auto()


class Layer(Layer):
    def _gen_args(self, input_shape, args):
        return {}

    def get_output_shape(self):
        from TensorNAS.Tools import dimension_mag

        return (1, dimension_mag(self.inputshape.get()))

    def get_keras_layers(self, input_tensor):
        import tensorflow as tf

        return tf.keras.layers.Flatten(input_shape=self.inputshape.get())(input_tensor)
