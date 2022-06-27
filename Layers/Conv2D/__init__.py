import random
import TensorNAS.Core.Layer
from TensorNAS.Core.Layer import (
    Layer,
    ArgActivations,
    ArgPadding,
    ArgInitializers,
    ArgRegularizers,
)
from TensorNAS.Core import EnumWithNone
from TensorNAS.Core.LayerMutations import (
    MutateFilters,
    MutateKernelSize,
    MutatePadding,
    MutateActivation,
    MutateDilationRate,
    MutateStrides,
)
from enum import auto


class Args(EnumWithNone):
    "Args needed for creating Conv2DArgs layer, list not complete"

    FILTERS = auto()
    KERNEL_SIZE = auto()
    STRIDES = auto()
    PADDING = auto()
    DILATION_RATE = auto()
    ACTIVATION = auto()
    GROUPS = auto()
    REGULARIZER = auto()
    INITIALIZER = auto()


class Layer(
    Layer,
    MutateFilters,
    MutateKernelSize,
    MutatePadding,
    MutateActivation,
    MutateDilationRate,
    MutateStrides,
):

    MAX_FILTER_COUNT = 128
    MAX_KERNEL_DIMENSION = 7
    MAX_STRIDE = 7
    MAX_DILATION = 5

    def _gen_args(self, input_shape, args):
        filter_count = random.randint(1, self.MAX_FILTER_COUNT)
        kernel_size = TensorNAS.Core.Layer.gen_2d_kernel_size(self.MAX_KERNEL_DIMENSION)
        padding = ArgPadding.SAME
        # Since Relu is the standard activation, we will start with Relu and let the EA mutate it
        activation = ArgActivations.NONE
        dilation_rate = (1, 1)
        initializer = ArgInitializers.GLOROT_UNIFORM
        regularizer = ArgRegularizers.NONE, 0
        strides = (1, 1)

        if args:
            if self.get_args_enum().FILTERS in args:
                filter_count = args.get(self.get_args_enum().FILTERS)
            if self.get_args_enum().KERNEL_SIZE in args:
                kernel_size = args.get(self.get_args_enum().KERNEL_SIZE)
            if self.get_args_enum().PADDING in args:
                padding = ArgPadding(args.get(self.get_args_enum().PADDING))
            if self.get_args_enum().ACTIVATION in args:
                activation = ArgActivations(args.get(self.get_args_enum().ACTIVATION))
            if self.get_args_enum().DILATION_RATE in args:
                dilation_rate = args.get(self.get_args_enum().DILATION_RATE)
            if self.get_args_enum().STRIDES in args:
                strides = args.get(self.get_args_enum().STRIDES)
            if self.get_args_enum().REGULARIZER in args:
                regularizer = args.get(self.get_args_enum().REGULARIZER)
            if self.get_args_enum().INITIALIZER in args:
                initializer = args.get(self.get_args_enum().INITIALIZER)

        return {
            self.get_args_enum().FILTERS: filter_count,
            self.get_args_enum().KERNEL_SIZE: kernel_size,
            self.get_args_enum().STRIDES: strides,
            self.get_args_enum().PADDING: padding,
            self.get_args_enum().DILATION_RATE: dilation_rate,
            self.get_args_enum().ACTIVATION: activation,
            self.get_args_enum().REGULARIZER: regularizer,
            self.get_args_enum().INITIALIZER: initializer,
        }

    @staticmethod
    def _same_pad_output_shape(input, stride):
        return ((int(input) - 1) // int(stride)) + 1

    @staticmethod
    def _valid_pad_output_shape(input, kernel, stride):
        return ((input - kernel) // stride) + 1

    @staticmethod
    def conv2Doutputshape(input_size, stride, kernel_size, filter_count, padding):
        if padding == TensorNAS.Core.Layer.ArgPadding.SAME:
            X = Layer._same_pad_output_shape(input_size[0], stride[0])
            Y = Layer._same_pad_output_shape(input_size[1], stride[1])
            return (int(X), int(Y), int(filter_count))
        elif padding == TensorNAS.Core.Layer.ArgPadding.VALID:
            X = Layer._valid_pad_output_shape(input_size[0], kernel_size[0], stride[0])
            Y = Layer._valid_pad_output_shape(input_size[1], kernel_size[1], stride[1])
            return (int(X), int(Y), int(filter_count))
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
