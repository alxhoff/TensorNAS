from TensorNAS.Core import EnumWithNone
from TensorNAS.Core.Layer import Layer
from TensorNAS.Core.LayerMutations import MutateStrides, MutatePadding, MutatePoolSize
from enum import auto


def valid_pad_output_shape(input, pool, stride):
    return int(((input - pool) // stride) + 1)


def same_pad_output_shape(input, pool, stride):
    return int(
        valid_pad_output_shape(input, pool, stride)
        + (1 if ((input - pool) % stride) else 0)
    )


class Args(EnumWithNone):

    POOL_SIZE = auto()
    STRIDES = auto()
    PADDING = auto()


class Layer(Layer, MutateStrides, MutatePadding, MutatePoolSize):
    MAX_POOL_SIZE = 7
    MAX_STRIDE = 7
