from math import ceil

import tensorflow as tf

from tensornas.core.layerargs import *
from tensornas.layers.MaxPool import Layer


def _valid_pad_output_shape(input, pool, stride):
    return ((input - pool) // stride) + 1


def _same_pad_output_shape(input, pool, stride):
    return _valid_pad_output_shape(input, pool, stride) + (
        1 if ((input - pool) % stride) else 0
    )


class Layer(Layer):
    def _gen_args(cls, input_shape, args):
        return {
            cls.get_args_enum().POOL_SIZE: gen_poolsize(ceil(input_shape[0] / 2)),
            cls.get_args_enum().STRIDES: gen_2d_strides(ceil(input_shape[0] / 2)),
            cls.get_args_enum().PADDING: gen_padding(),
        }

    def repair(self):
        for x, val in enumerate(self.args[self.get_args_enum().STRIDES]):
            if not val > 0:
                self.args[self.get_args_enum().STRIDES][x] = 1

        for x, val in enumerate(self.args[self.get_args_enum().POOL_SIZE]):
            if not val > 0:
                self.args[self.get_args_enum().POOL_SIZE][x] = 1

    def get_output_shape(self):
        inp = self.inputshape.get()
        pool = self.args[self.get_args_enum().POOL_SIZE]
        stri = self.args[self.get_args_enum().STRIDES]
        pad = self.args[self.get_args_enum().PADDING]
        if pad == ArgPadding.SAME:
            x = _same_pad_output_shape(inp[0], pool[0], stri[0])
            y = _same_pad_output_shape(inp[1], pool[1], stri[1])
            try:
                return (x, y, inp[2])
            except Exception as e:
                raise Exception("I/O shapes not able to be made compatible")
        elif pad == ArgPadding.VALID:
            x = _valid_pad_output_shape(inp[0], pool[0], stri[0])
            y = _valid_pad_output_shape(inp[1], pool[1], stri[1])
            try:
                return (x, y, inp[2])
            except Exception as e:
                raise Exception("I/O shapes not able to be made compatible")
        return (0, 0, 0)

    def get_keras_layer(self):
        return tf.keras.layers.MaxPool2D(
            input_shape=self.inputshape.get(),
            pool_size=self.args.get(self.get_args_enum().POOL_SIZE),
            strides=self.args.get(self.get_args_enum().STRIDES),
            padding=self.args.get(self.get_args_enum().PADDING).value,
        )
