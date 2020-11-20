import random
from math import ceil
import tensorflow as tf

from tensornas.layers.Conv2D import Layer
import tensornas.core.layerargs as la
from tensornas.core.util import MutationOperators


class Layer(Layer):
    def _gen_args(cls, input_shape, args):
        return {
            cls.get_args_enum().FILTERS: random.randint(1, ceil(input_shape[0] / 2)),
            cls.get_args_enum().KERNEL_SIZE: (1, 1),
            cls.get_args_enum().STRIDES: (1, 1),
            cls.get_args_enum().PADDING: la.gen_padding(),
            cls.get_args_enum().DILATION_RATE: la.gen_dilation(),
            cls.get_args_enum().ACTIVATION: la.gen_activation(),
        }

    def _mutate_kernel_size(self, operator=MutationOperators.SYNC_STEP):
        """
        A pointwise conv requires a 1x1 kernel, thus we cannot mutate it. This function is here to override the
        standard conv mutations found in the super class.
        """
        return

    def get_keras_layer(self):
        return [
            tf.keras.layers.Conv2D(
                filters=self.args.get(self.get_args_enum().FILTERS),
                kernel_size=self.args.get(self.get_args_enum().KERNEL_SIZE),
                strides=self.args.get(self.get_args_enum().STRIDES),
                input_shape=self.inputshape.get(),
                activation=self.args.get(self.get_args_enum().ACTIVATION).value,
                padding=self.args.get(self.get_args_enum().PADDING).value,
                dilation_rate=self.args.get(self.get_args_enum().DILATION_RATE),
            )
        ]
