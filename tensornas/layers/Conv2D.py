import random
from enum import Enum, auto
from math import ceil

import tensorflow as tf

import tensornas.core.layerargs as la
from tensornas.core.layer import NetworkLayer
from tensornas.core.util import mutate_int, mutate_enum, mutate_tuple, MutationOperators


class Args(Enum):
    "Args needed for creating Conv2DArgs layer, list not complete"
    FILTERS = auto()
    KERNEL_SIZE = auto()
    STRIDES = auto()
    PADDING = auto()
    DILATION_RATE = auto()
    ACTIVATION = auto()


class Layer(NetworkLayer):
    MAX_FILTER_COUNT = 128
    MAX_KERNEL_DIMENSION = 7
    MAX_STRIDE = 7
    MAX_DILATION = 5

    def _gen_args(cls, input_shape, args):
        return {
            cls.get_args_enum().FILTERS: random.randint(1, cls.MAX_FILTER_COUNT),
            cls.get_args_enum().KERNEL_SIZE: la.gen_kernel_size(
                cls.MAX_KERNEL_DIMENSION
            ),
            cls.get_args_enum().STRIDES: (1, 1),
            cls.get_args_enum().PADDING: la.gen_padding(),
            cls.get_args_enum().DILATION_RATE: la.gen_dilation(),
            cls.get_args_enum().ACTIVATION: la.gen_activation(),
        }

    def _mutate_filters(self, operator=MutationOperators.STEP):
        self.args[self.get_args_enum().FILTERS] = mutate_int(
            self.args[self.get_args_enum().FILTERS], 1, self.MAX_FILTER_COUNT, operator,
        )

    def _mutate_kernel_size(self, operator=MutationOperators.SYNC_STEP):
        self.args[self.get_args_enum().KERNEL_SIZE] = mutate_tuple(
            self.args[self.get_args_enum().KERNEL_SIZE],
            1,
            self.MAX_KERNEL_DIMENSION,
            operator,
        )

    def _mutate_strides(self, operator=MutationOperators.SYNC_STEP):
        self.args[self.get_args_enum().STRIDES] = mutate_tuple(
            self.args[self.get_args_enum().STRIDES], 1, self.MAX_STRIDE, operator,
        )

    def _mutate_padding(self):
        self.args[self.get_args_enum().PADDING] = mutate_enum(
            self.args[self.get_args_enum().PADDING], la.ArgPadding
        )

    def _mutate_dilation_rate(self, operator=MutationOperators.SYNC_STEP):
        self.args[self.get_args_enum().DILATION_RATE] = mutate_tuple(
            self.args[self.get_args_enum().DILATION_RATE],
            1,
            self.MAX_DILATION,
            operator,
        )

    def _mutate_activation(self):
        self.args[self.get_args_enum().ACTIVATION] = mutate_enum(
            self.args[self.get_args_enum().ACTIVATION], la.ArgActivations
        )

    def _single_stride(self):
        st = self.args[self.get_args_enum().STRIDES]
        if st[0] == 1 and st[1] == 1:
            return True
        return False

    def _single_dilation_rate(self):
        dr = self.args[self.get_args_enum().DILATION_RATE]
        if dr[0] == 1 and dr[1]:
            return True
        return False

    def validate(self, repair=True):
        if not self.args[self.get_args_enum().FILTERS] > 0:
            return False

        if not self._single_stride() and not self._single_dilation_rate():
            return False

        if (
            (self.inputshape.get()[0] - self.args[self.get_args_enum().KERNEL_SIZE][0])
            < 1
        ) or (
            (self.inputshape.get()[1] - self.args[self.get_args_enum().KERNEL_SIZE][1])
            < 1
        ):
            return False

        if (
            not self.args[self.get_args_enum().STRIDES][0] > 0
            or not self.args[self.get_args_enum().STRIDES][1] > 0
        ):
            return False

        return True

    @staticmethod
    def _valid_pad_output_shape(input, kernel, stride):
        return ((input - kernel) // stride) + 1

    @staticmethod
    def _same_pad_output_shape(input, stride):
        return ((input - 1) // stride) + 1

    @staticmethod
    def conv2Doutputshape(input_size, stride, kernel_size, filter_count, padding):
        if padding == la.ArgPadding.SAME:
            X = Layer._same_pad_output_shape(input_size[0], stride[0])
            Y = Layer._same_pad_output_shape(input_size[1], stride[1])
            return (X, Y, filter_count)
        elif padding == la.ArgPadding.VALID:
            X = Layer._valid_pad_output_shape(input_size[0], kernel_size[0], stride[0])
            Y = Layer._valid_pad_output_shape(input_size[1], kernel_size[1], stride[1])
            return (X, Y, filter_count)
        else:
            raise Exception("Invalid Conv2D padding for calculating output shape")

    def get_output_shape(self):
        return Layer.conv2Doutputshape(
            input_size=self.inputshape.get(),
            stride=self.args[self.get_args_enum().STRIDES],
            kernel_size=self.args[self.get_args_enum().KERNEL_SIZE],
            filter_count=self.args[self.get_args_enum().FILTERS],
            padding=self.args[self.get_args_enum().PADDING],
        )

    def get_keras_layer(self):
        return [
            tf.keras.layers.Conv2D(
                filters=self.args.get(self.get_args_enum().FILTERS),
                kernel_size=self.args.get(self.get_args_enum().KERNEL_SIZE),
                strides=self.args.get(self.get_args_enum().STRIDES),
                input_shape=self.inputshape.get(),
                activation=self.args.get(self.get_args_enum().ACTIVATION).value,
                padding=self.args.get(self.get_args_enum().PADDING).value,
                dilation_rate=self.args.get(self.get_args_enum().DILATION_RATE),
            )
        ]
