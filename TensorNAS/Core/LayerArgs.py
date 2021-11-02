import random

from TensorNAS.Layers import *


class ArgActivations(str, Enum):
    ELU = "elu"
    EXPONENTIAL = "exponential"
    HARD_SIGMOID = "hard_sigmoid"
    LINEAR = "linear"
    RELU = "relu"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    SOFTPLUS = "softplus"
    TANH = "tanh"


class ArgPadding(str, Enum):
    VALID = "valid"
    SAME = "same"


class ArgRegularizers(str, Enum):
    L1 = "L1"
    L1L2 = "L1L2"
    L2 = "L2"


def gen_2d_kernel_size(input_size):
    kernel_size = random.choice(range(1, input_size, 2))
    return (kernel_size, kernel_size)


def gen_3d_strides(max_bound):
    stride_size = random.randint(1, max_bound)
    return (stride_size, stride_size, stride_size)


def gen_2d_strides(max_bound):
    stride_size = random.randint(1, max_bound)
    return (stride_size, stride_size)


def gen_1d_strides(max_bound):
    return random.randint(1, max_bound)


def gen_3d_poolsize(max_bound):
    size = random.randint(1, max_bound)
    return (size, size, size)


def gen_2d_poolsize(max_bound):
    size = random.randint(1, max_bound)
    return (size, size)


def gen_1d_poolsize(max_bound):
    return random.randint(1, max_bound)


def gen_2d_dilation():
    # TODO
    return (1, 1)


def gen_dropout(max):
    while True:
        ret = round(random.uniform(0, max), 2)
        if ret != 0.0:
            return ret


def gen_padding():
    return random.choice(list(ArgPadding))


def gen_activation():
    return random.choice(list(ArgActivations))


def gen_groups(max_bound):
    return random.randint(1, max_bound)
