import tensorflow as tf

from tensornas.core.layerargs import *
from tensornas.layers.MaxPool import (
    Layer,
    valid_pad_output_shape,
    same_pad_output_shape,
)
from tensornas.core.util import MutationOperators


class Layer(Layer):
    MAX_POOL_SIZE = 5
    MAX_STRIDE_SIZE = 5

    def _gen_args(self, input_shape, args):
        return {
            self.get_args_enum().POOL_SIZE: gen_1d_poolsize(
                random.randint(1, self.MAX_POOL_SIZE)
            ),
            self.get_args_enum().STRIDES: gen_1d_strides(
                random.randint(1, self.MAX_STRIDE_SIZE)
            ),
            self.get_args_enum().PADDING: gen_padding(),
        }

    def get_output_shape(self):
        inp = self.inputshape.get()
        pool = self.args[self.get_args_enum().POOL_SIZE]
        stri = self.args[self.get_args_enum().STRIDES]
        pad = self.args[self.get_args_enum().PADDING]
        if pad == ArgPadding.SAME:
            x = same_pad_output_shape(inp[0], pool, stri)
            y = same_pad_output_shape(inp[1], pool, stri)
            try:
                return (x, y)
            except Exception as e:
                raise Exception("I/O shapes not able to be made compatible")
        elif pad == ArgPadding.VALID:
            x = valid_pad_output_shape(inp[0], pool, stri)
            y = valid_pad_output_shape(inp[1], pool, stri)
            try:
                return (x, y)
            except Exception as e:
                raise Exception("I/O shapes not able to be made compatible")
        return (0, 0)

    def get_keras_layer(self, input_tensor):
        return tf.keras.layers.MaxPool1D(
            input_shape=self.inputshape.get(),
            pool_size=self.args.get(self.get_args_enum().POOL_SIZE),
            strides=self.args.get(self.get_args_enum().STRIDES),
            padding=self.args.get(self.get_args_enum().PADDING).value,
        )(input_tensor)
