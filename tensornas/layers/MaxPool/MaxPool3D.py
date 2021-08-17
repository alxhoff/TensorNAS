from math import ceil


from tensornas.core.layerargs import *
from tensornas.layers.MaxPool import Layer


class Layer(Layer):
    def _gen_args(self, input_shape, args):
        return {
            self.get_args_enum().POOL_SIZE: gen_3d_poolsize(
                random.randint(1, self.MAX_POOL_SIZE)
            ),
            self.get_args_enum().STRIDES: gen_3d_strides(
                random.randint(1, self.MAX_STRIDE_SIZE)
            ),
            self.get_args_enum().PADDING: gen_padding(),
        }

    def get_output_shape(self):
        # TODO
        return self.inputshape.get()

    def get_keras_layer(self, input_tensor):
        import tensorflow as tf

        return tf.keras.layers.MaxPool3D(
            input_shape=self.inputshape.get(),
            pool_size=self.args.get(self.get_args_enum().POOL_SIZE),
            strides=self.args.get(self.get_args_enum().STRIDES),
            padding=self.args.get(self.get_args_enum().PADDING).value,
        )(input_tensor)
