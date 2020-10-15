from math import ceil
import tensorflow as tf

from tensornas.core.layerargs import *
from tensornas.layers.MaxPool import Layer


class Layer(Layer):
    def _gen_args(self, input_shape, args):
        return {
            self.get_args_enum().POOL_SIZE.value: gen_poolsize(
                ceil(input_shape[0] / 2)
            ),
            self.get_args_enum().STRIDES.value: gen_3d_strides(
                ceil(input_shape[0] / 2)
            ),
            self.get_args_enum().PADDING.value: gen_padding(),
        }

    def get_output_shape(self):
        # TODO
        return self.inputshape.get()

    def get_keras_layer(self):
        return tf.keras.layers.MaxPool3D(
            pool_size=self.args.get(self.get_args_enum().POOL_SIZE.value),
            strides=self.args.get(self.get_args_enum().STRIDES.value),
            padding=self.args.get(self.get_args_enum().PADDING.value),
        )
