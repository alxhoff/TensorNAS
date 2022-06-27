from TensorNAS.Layers.Pool import Layer


class Layer(Layer):
    MAX_POOL_SIZE = 5
    MAX_STRIDE_SIZE = 5

    def _gen_args(self, input_shape, args):
        from TensorNAS.Core.Layer import gen_padding
        from TensorNAS.Core.Layer import gen_2d_poolsize
        from TensorNAS.Core.Layer import gen_2d_strides
        import random

        pool_size = gen_2d_poolsize(random.randint(1, self.MAX_POOL_SIZE))
        stride_size = gen_2d_strides(random.randint(1, self.MAX_STRIDE_SIZE))
        padding = gen_padding()

        if args:
            if self.get_args_enum().PADDING in args:
                from TensorNAS.Core.Layer import ArgPadding

                padding = ArgPadding(args.get(self.get_args_enum().PADDING))
            if self.get_args_enum().STRIDES in args:
                stride_size = args.get(self.get_args_enum().STRIDES)
            if self.get_args_enum().POOL_SIZE in args:
                pool_size = args.get(self.get_args_enum().POOL_SIZE)

        return {
            self.get_args_enum().POOL_SIZE: pool_size,
            self.get_args_enum().STRIDES: stride_size,
            self.get_args_enum().PADDING: padding,
        }

    def repair(self):
        for x, val in enumerate(self.args[self.get_args_enum().STRIDES]):
            if not val > 0:
                self.args[self.get_args_enum().STRIDES][x] = 1

        for x, val in enumerate(self.args[self.get_args_enum().POOL_SIZE]):
            if not val > 0:
                self.args[self.get_args_enum().POOL_SIZE][x] = 1

    def get_output_shape(self):
        from TensorNAS.Core.Layer import ArgPadding
        from TensorNAS.Layers.Pool import (
            valid_pad_output_shape,
            same_pad_output_shape,
        )

        inp = self.inputshape.get()
        pool = self.args[self.get_args_enum().POOL_SIZE]
        stri = self.args[self.get_args_enum().STRIDES]

        if stri is None:
            stri = pool

        pad = self.args[self.get_args_enum().PADDING]
        if pad == ArgPadding.SAME:
            x = same_pad_output_shape(inp[0], pool[0], stri[0])
            y = same_pad_output_shape(inp[1], pool[1], stri[1])
            try:
                return int(x), int(y), int(inp[2])
            except Exception as e:
                raise Exception("I/O shapes not able to be made compatible")
        elif pad == ArgPadding.VALID:
            x = valid_pad_output_shape(inp[0], pool[0], stri[0])
            y = valid_pad_output_shape(inp[1], pool[1], stri[1])
            try:
                return int(x), int(y), int(inp[2])
            except Exception as e:
                raise Exception("I/O shapes not able to be made compatible")
        return 0, 0, 0

    def get_keras_layers(self, input_tensor):
        import tensorflow as tf

        return tf.keras.layers.MaxPool2D(
            input_shape=self.inputshape.get(),
            pool_size=self.args.get(self.get_args_enum().POOL_SIZE),
            strides=self.args.get(self.get_args_enum().STRIDES),
            padding=self.args.get(self.get_args_enum().PADDING).value(),
        )(input_tensor)
