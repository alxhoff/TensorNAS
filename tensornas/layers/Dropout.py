from enum import auto

import tensorflow as tf

import tensornas.core.layerargs as la
from tensornas.core.layer import NetworkLayer
from tensornas.core.layerargs import *
from tensornas.core.util import mutate_unit_interval


class Args(Enum):
    "Args needed for creating Dropout layer, list not complete"
    RATE = auto()


class Layer(NetworkLayer):
    MAX_RATE = 0.5

    def _gen_args(cls, input_shape, max):
        if not max:
            max = 1.0
        return {cls.get_args_enum().RATE: la.gen_dropout(min(max, cls.MAX_RATE))}

    def _mutate_rate(self):
        self.args[self.get_args_enum().RATE] = mutate_unit_interval(
            self.args[self.get_args_enum().RATE], 0, self.MAX_RATE
        )

    def get_output_shape(self):
        return self.inputshape.get()

    def get_keras_layer(self):
        return tf.keras.layers.Dropout(
            rate=self.args.get(self.get_args_enum().RATE),
            input_shape=self.inputshape.get(),
        )
