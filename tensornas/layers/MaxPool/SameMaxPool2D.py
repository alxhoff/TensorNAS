from tensornas.layers.MaxPool.MaxPool2D import *
from tensornas.layers.MaxPool import Args

import tensornas.core.layerargs as la


class Layer(Layer):
    def _gen_args(self, input_shape, args):
        return {
            self.get_args_enum().POOL_SIZE: gen_2d_poolsize(
                random.randint(1, self.MAX_POOL_SIZE)
            ),
            self.get_args_enum().STRIDES: (1, 1),
            self.get_args_enum().PADDING: la.ArgPadding.SAME,
        }
