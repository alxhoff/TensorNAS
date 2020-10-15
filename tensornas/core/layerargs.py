import random
from enum import Enum
from tensornas.layers import *


class ArgActivations(str, Enum):
    ELU = "elu"
    EXPONENTIAL = "exponential"
    HARD_SIGMOID = "hard_sigmoid"
    LINEAR = "linear"
    RELU = "relu"
    SELU = "selu"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"
    SOFTPLUS = "softplus"
    SOFTSIGN = "softsign"
    SWISH = "swish"
    TANH = "tanh"


class ArgPadding(str, Enum):
    VALID = "valid"
    SAME = "same"


def gen_kernel_size(input_size):
    kernel_size = random.randint(1, input_size)
    return (kernel_size, kernel_size)


def gen_2d_strides(max_bound):
    stride_size = random.randint(1, max_bound)
    return (stride_size, stride_size)


def gen_3d_strides(max_bound):
    stride_size = random.randint(1, max_bound)
    return (stride_size, stride_size, stride_size)


def gen_poolsize(max_bound):
    size = random.randint(1, max_bound)
    return (size, size)


def gen_dilation():
    # TODO
    return (1, 1)


def gen_dropout(max):
    while True:
        ret = round(random.uniform(0, max), 2)
        if ret != 0.0:
            return ret


def gen_padding():
    return random.choice(list(ArgPadding)).value


def gen_activation():
    return random.choice(list(ArgActivations)).value


def create_layer_args(layer_type, input_shape, args):
    """
    Each layer type gets a function of the form 'gen_' + LayerType enum value, ie. string name of layer, + '_args'.
    The function must take in two arguments, the input shape and args. Args is used to pass in layer specific values
    eg. A classification dense layer will need to know how many classes to have while other layers have no need of this
    argument. Since eval is used to call these functions the arguments must be provided in the function definition, even
    if not required.
    """
    return eval("gen_" + layer_type.value + "_args")(input_shape, args)
