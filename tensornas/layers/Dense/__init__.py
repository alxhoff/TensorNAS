from enum import Enum, auto

import tensorflow as tf

from tensornas.core.layer import NetworkLayer


class Args(Enum):
    "Args needed for creating Dense layer, list not complete"
    UNITS = auto()
    ACTIVATION = auto()


class Layer(NetworkLayer):
    def validate(self, repair=True):
        # Dense layers take in a 1D tensor array, ie. previous layer should be a flatten layer
        if not len(self.inputshape.get()) == 1:
            return False

        return True

    def get_output_shape(self):
        return (1, self.args.get(self.get_args_enum().UNITS))

    def get_keras_layer(self):
        return tf.keras.layers.Dense(
            self.args.get(self.get_args_enum().UNITS),
            activation=self.args.get(self.get_args_enum().ACTIVATION).value,
        )
