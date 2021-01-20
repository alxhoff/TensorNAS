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

    def _gen_args(self, input_shape, args):

        max = la.gen_dropout(self.MAX_RATE)

        if args:
            if self.get_args_enum().RATE in args:
                max = args.get(self.get_args_enum().RATE)

        return {self.get_args_enum().RATE: max}

    def _mutate_rate(self):
        self.args[self.get_args_enum().RATE] = mutate_unit_interval(
            self.args[self.get_args_enum().RATE], 0, self.MAX_RATE
        )

    def get_output_shape(self):
        return self.inputshape.get()

    def get_keras_layer(self, input_tensor):
        return tf.keras.layers.Dropout(
            rate=self.args.get(self.get_args_enum().RATE),
            input_shape=self.inputshape.get(),
        )(input_tensor)
