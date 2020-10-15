from enum import Enum, auto
import tensorflow as tf

from tensornas.core.networklayer import NetworkLayer
from tensornas.core.layerargs import *
from tensornas.core.util import mutate_unit_interval
import tensornas.core.layerargs as la


class Args(Enum):
    "Args needed for creating Dropout layer, list not complete"
    RATE = auto()


class Layer(NetworkLayer):
    MAX_RATE = 0.5

    def _mutate_rate(self):
        self.args[self.get_args_enum().RATE.value] = mutate_unit_interval(
            self.args[self.get_args_enum().RATE.value], 0, self.MAX_RATE
        )

    def _gen_args(self, input_shape, max):
        if not max:
            max = 1.0
        return {
            self.get_args_enum().RATE.value: la.gen_dropout(min(max, self.MAX_RATE))
        }

    def get_output_shape(self):
        return self.inputshape.get()

    def get_keras_layer(self):
        return tf.keras.layers.Dropout(
            rate=self.args.get(self.get_args_enum().RATE.value)
        )
