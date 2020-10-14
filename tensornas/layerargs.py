from enum import Enum, auto
import random
from math import ceil


class Activations(str, Enum):
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


class PaddingArgs(str, Enum):
    VALID = "valid"
    SAME = "same"


class Conv2DArgs(Enum):
    "Args needed for creating Conv2DArgs layer, list not complete"
    FILTERS = auto()
    KERNEL_SIZE = auto()
    STRIDES = auto()
    PADDING = auto()
    DILATION_RATE = auto()
    ACTIVATION = auto()


class MaxPool2DArgs(Enum):
    "Args needed for creating MaxPool2D layer, list not complete"
    POOL_SIZE = auto()
    STRIDES = auto()
    PADDING = auto()


class ReshapeArgs(Enum):
    "Args needed for creating Reshape layer"
    TARGET_SHAPE = auto()


class DenseArgs(Enum):
    "Args needed for creating Dense layer, list not complete"
    UNITS = auto()
    ACTIVATION = auto()


class DropoutArgs(Enum):
    "Args needed for creating Dropout layer, list not complete"
    RATE = auto()


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


def gen_padding():
    return random.choice(list(PaddingArgs)).value


def gen_activation():
    return random.choice(list(Activations)).value


"""
Each layer type gets a function of the form 'gen_' + LayerType enum value, ie. string name of layer, + '_args'.
The function must take in two arguments, the input shape and args. Args is used to pass in layer specific values
eg. A classification dense layer will need to know how many classes to have while other layers have no need of this
argument. Since eval is used to call these functions the arguments must be provided in the function definition, even
if not required. 
"""


def gen_Conv2D_args(input_shape, args):
    try:
        return {
            Conv2DArgs.FILTERS.value: random.randint(1, ceil(input_shape[0] / 2)),
            Conv2DArgs.KERNEL_SIZE.value: gen_kernel_size(ceil(input_shape[0] / 2)),
            Conv2DArgs.STRIDES.value: [1, 1],
            Conv2DArgs.PADDING.value: gen_padding(),
            Conv2DArgs.DILATION_RATE.value: gen_dilation(),
        }
    except Exception as e:
        print(e)


def gen_MaxPool2D_args(input_shape, args):
    return {
        MaxPool2DArgs.POOL_SIZE.value: gen_poolsize(ceil(input_shape[0] / 2)),
        MaxPool2DArgs.STRIDES.value: gen_2d_strides(ceil(input_shape[0] / 2)),
        MaxPool2DArgs.PADDING.value: gen_padding(),
    }


def gen_MaxPool3D_args(input_shape, args):
    return {
        MaxPool2DArgs.POOL_SIZE.value: gen_poolsize(ceil(input_shape[0] / 2)),
        MaxPool2DArgs.STRIDES.value: gen_3d_strides(ceil(input_shape[0] / 2)),
        MaxPool2DArgs.PADDING.value: gen_padding(),
    }


def gen_Dense_args(input_shape, args):
    if args:
        units = args
        act = "softmax"
    else:
        units = random.randint(1, 512)
        act = gen_activation()
    return {DenseArgs.UNITS.value: units, DenseArgs.ACTIVATION.value: act}


def gen_Flatten_args(input_shape, none):
    return {}


def gen_Dropout_args(input_shape, max=1.0):
    return {DropoutArgs.RATE: random.uniform(0, max)}


def gen_Reshape_args(input_shape, target_shape):
    return {ReshapeArgs.TARGET_SHAPE.value: target_shape}


def create_layer_args(layer_type, input_shape, args):
    return eval("gen_" + layer_type.value + "_args")(input_shape, args)
