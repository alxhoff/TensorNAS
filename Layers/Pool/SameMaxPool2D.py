import random
from TensorNAS.Layers.Pool.MaxPool2D import Layer
from TensorNAS.Core.Layer import gen_2d_poolsize
from TensorNAS.Core.Layer import ArgPadding as ap


class Layer(Layer):
    def _gen_args(self, input_shape, args):

        pool_size = gen_2d_poolsize(random.randint(1, self.MAX_POOL_SIZE))
        stride_size = (1, 1)
        padding = ap.SAME

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
