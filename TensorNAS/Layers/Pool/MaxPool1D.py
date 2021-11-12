from TensorNAS.Core.Layer import (
    ArgPadding,
    gen_1d_strides,
    gen_1d_poolsize,
    gen_padding,
)
from TensorNAS.Layers.Pool import (
    Layer,
    valid_pad_output_shape,
    same_pad_output_shape,
)


class Layer(Layer):
    MAX_POOL_SIZE = 5
    MAX_STRIDE_SIZE = 5

    def _gen_args(self, input_shape, args):
        import random

        pool_size = gen_1d_poolsize(random.randint(1, self.MAX_POOL_SIZE))
        max_stride_size = gen_1d_strides(random.randint(1, self.MAX_STRIDE_SIZE))
        padding = gen_padding()

        if self.get_args_enum().POOL_SIZE in args:
            pool_size = args.get(self.get_args_enum().POOL_SIZE)
        if self.get_args_enum().STRIDES in args:
            max_stride_size = args.get(self.get_args_enum().STRIDES)
        if self.get_args_enum().PADDING in args:
            from TensorNAS.Core.Layer import ArgPadding

            padding = ArgPadding(args.get(self.get_args_enum().PADDING))

        return {
            self.get_args_enum().POOL_SIZE: pool_size,
            self.get_args_enum().STRIDES: max_stride_size,
            self.get_args_enum().PADDING: padding,
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

    def get_keras_layers(self, input_tensor):
        import tensorflow as tf

        return tf.keras.layers.MaxPool1D(
            input_shape=self.inputshape.get(),
            pool_size=self.args.get(self.get_args_enum().POOL_SIZE),
            strides=self.args.get(self.get_args_enum().STRIDES),
            padding=self.args.get(self.get_args_enum().PADDING).value(),
        )(input_tensor)
