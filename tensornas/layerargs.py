from enum import Enum, auto


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
    INPUT_SHAPE = auto()
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
