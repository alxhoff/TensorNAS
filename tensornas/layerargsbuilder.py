import random
from tensornas.layerargs import *


def get_kernel_size(input_size):
    kernel_size = random.randint(1, input_size)
    return [kernel_size, kernel_size]


def get_2d_strides(max_bound):
    stride_size = random.randint(1, max_bound)
    return [stride_size, stride_size]


def get_3d_strides(max_bound):
    stride_size = random.randint(1, max_bound)
    return [stride_size, stride_size, stride_size]


def gen_poolsize(max_bound):
    size = random.randint(1, max_bound)
    return [size, size]


def gen_conv2d_args(input_shape):
    return {
        Conv2DArgs.INPUT_SHAPE.name: input_shape,
        Conv2DArgs.FILTERS.name: random.randint(1, input_shape[0] / 2),
        Conv2DArgs.KERNEL_SIZE.name: get_kernel_size(input_shape[0] / 2),
        Conv2DArgs.STRIDES.name: [1, 1],
        Conv2DArgs.PADDING.name: "valid",
    }


def gen_pool2d_args(input_shape):
    return {
        MaxPool2DArgs.POOL_SIZE.name: gen_poolsize(input_shape[0] / 2),
        MaxPool2DArgs.STRIDES.name: get_2d_strides(input_shape[0] / 2),
    }


def gen_pool3d_args(input_shape):
    return {
        MaxPool2DArgs.POOL_SIZE.name: gen_poolsize(input_shape[0] / 2),
        MaxPool2DArgs.STRIDES.name: get_3d_strides(input_shape[0] / 2),
    }


def gen_dense_args(activation="relu", no_classes=None):
    if no_classes:
        units = no_classes
        act = "softmax"
    else:
        units = random.randint(1, 512)
        act = activation
    return {DenseArgs.UNITS.name: units, DenseArgs.ACTIVATION.name: act}


def gen_drop_args():
    return {DropoutArgs.RATE: random.random()}


def gen_reshape_args(target_shape):
    return {ReshapeArgs.TARGET_SHAPE.name: target_shape}
