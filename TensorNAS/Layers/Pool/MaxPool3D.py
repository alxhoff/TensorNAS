from TensorNAS.Core.Layer import gen_3d_strides, gen_3d_poolsize, gen_padding
from TensorNAS.Layers.Pool import Layer


class Layer(Layer):
    def _gen_args(self, input_shape, args):
        import random

        pool_size = gen_3d_poolsize(random.randint(1, self.MAX_POOL_SIZE))
        stride_size = gen_3d_strides(random.randint(1, self.MAX_STRIDE_SIZE))
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

    def get_output_shape(self):
        # TODO
        return self.inputshape.get()

    def get_keras_layers(self, input_tensor):
        import tensorflow as tf

        return tf.keras.layers.MaxPool3D(
            input_shape=self.inputshape.get(),
            pool_size=self.args.get(self.get_args_enum().POOL_SIZE),
            strides=self.args.get(self.get_args_enum().STRIDES),
            padding=self.args.get(self.get_args_enum().PADDING).value(),
        )(input_tensor)
