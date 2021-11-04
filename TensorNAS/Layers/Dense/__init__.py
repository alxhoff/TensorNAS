from enum import Enum, auto

import tensorflow as tf

from TensorNAS.Core.Layer import Layer


class Args(Enum):
    "Args needed for creating Dense layer, list not complete"
    UNITS = auto()
    ACTIVATION = auto()


class Layer(Layer):
    def get_output_shape(self):
        return (1, self.args.get(self.get_args_enum().UNITS))

    def get_keras_layers(self, input_tensor):
        return tf.keras.layers.Dense(
            units=self.args.get(self.get_args_enum().UNITS),
            activation=self.args.get(self.get_args_enum().ACTIVATION).value,
            input_shape=self.inputshape.get(),
        )(input_tensor)
