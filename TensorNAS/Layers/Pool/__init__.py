from enum import Enum, auto

import TensorNAS.Core.Layer
from TensorNAS.Core.Layer import Layer
from TensorNAS.Core.Mutate import mutate_tuple, mutate_enum, MutationOperators


def valid_pad_output_shape(input, pool, stride):
    return ((input - pool) // stride) + 1


def same_pad_output_shape(input, pool, stride):
    return valid_pad_output_shape(input, pool, stride) + (
        1 if ((input - pool) % stride) else 0
    )


class Args(Enum):
    "Args needed for creating MaxPool2D layer, list not complete"
    POOL_SIZE = auto()
    STRIDES = auto()
    PADDING = auto()


class Layer(Layer):
    MAX_POOL_SIZE = 7
    MAX_STRIDE = 7

    def _mutate_pool_size(self, operator=MutationOperators.SYNC_STEP):
        self.args[self.get_args_enum().POOL_SIZE] = mutate_tuple(
            self.args[self.get_args_enum().POOL_SIZE],
            1,
            self.MAX_POOL_SIZE,
            operator=operator,
        )

    def _mutate_strides(self, operator=MutationOperators.SYNC_STEP):
        self.args[self.get_args_enum().STRIDES] = mutate_tuple(
            self.args[self.get_args_enum().STRIDES],
            1,
            self.MAX_STRIDE,
            operator=operator,
        )

    def _mutate_padding(self):
        self.args[self.get_args_enum().PADDING] = mutate_enum(
            self.args[self.get_args_enum().PADDING], TensorNAS.Core.Layer.ArgPadding
        )
