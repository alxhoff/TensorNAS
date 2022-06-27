from TensorNAS.Layers.Conv2D import Layer
from enum import auto
from TensorNAS.Core import EnumWithNone


class Args(EnumWithNone):
    "Args needed for creating Conv2DArgs layer, list not complete"

    FILTERS = auto()
    KERNEL_SIZE = auto()
    STRIDES = auto()
    PADDING = auto()
    DILATION_RATE = auto()
    ACTIVATION = auto()
    GROUPS = auto()
    DEPTHWISE_REGULARIZER = auto()
    POINTWISE_REGULARIZER = auto()
    DEPTHWISE_INITIALIZER = auto()
    POINTWISE_INITIALIZER = auto()


class Layer(Layer):
    def _gen_args(self, input_shape, args):
        import random
        import TensorNAS.Core.Layer
        from TensorNAS.Core.Layer import (
            ArgActivations,
            ArgPadding,
            ArgInitializers,
            ArgRegularizers,
        )

        filter_count = random.randint(1, self.MAX_FILTER_COUNT)
        kernel_size = TensorNAS.Core.Layer.gen_2d_kernel_size(self.MAX_KERNEL_DIMENSION)
        padding = ArgPadding.VALID
        # Since Relu is the standard activation, we will start with Relu and let the EA mutate it
        activation = ArgActivations.NONE
        dilation_rate = (1, 1)
        depthwise_initializer = ArgInitializers.GLOROT_UNIFORM
        pointwise_initializer = ArgInitializers.GLOROT_UNIFORM
        depthwise_regularizer = (ArgRegularizers.NONE, 0)
        pointwise_regularizer = (ArgRegularizers.NONE, 0)
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
            if self.get_args_enum().DEPTHWISE_REGULARIZER in args:
                depthwise_regularizer = args.get(
                    self.get_args_enum().DEPTHWISE_REGULARIZER
                )
            if self.get_args_enum().POINTWISE_REGULARIZER in args:
                pointwise_regularizer = args.get(
                    self.get_args_enum().POINTWISE_REGULARIZER
                )
            if self.get_args_enum().DEPTHWISE_INITIALIZER in args:
                depthwise_initializer = args.get(
                    self.get_args_enum().DEPTHWISE_INITIALIZER
                )
            if self.get_args_enum().POINTWISE_INITIALIZER in args:
                pointwise_initializer = args.get(
                    self.get_args_enum().POINTWISE_INITIALIZER
                )

        return {
            self.get_args_enum().FILTERS: filter_count,
            self.get_args_enum().KERNEL_SIZE: kernel_size,
            self.get_args_enum().STRIDES: strides,
            self.get_args_enum().PADDING: padding,
            self.get_args_enum().DILATION_RATE: dilation_rate,
            self.get_args_enum().ACTIVATION: activation,
            self.get_args_enum().DEPTHWISE_REGULARIZER: depthwise_regularizer,
            self.get_args_enum().POINTWISE_REGULARIZER: pointwise_regularizer,
            self.get_args_enum().DEPTHWISE_INITIALIZER: depthwise_initializer,
            self.get_args_enum().POINTWISE_INITIALIZER: pointwise_initializer,
        }

    def get_keras_layers(self, input_tensor):
        import tensorflow as tf
        from TensorNAS.Core.Layer import gen_regularizer

        return tf.keras.layers.SeparableConv2D(
            filters=self.args.get(self.get_args_enum().FILTERS),
            kernel_size=self.args.get(self.get_args_enum().KERNEL_SIZE),
            strides=self.args.get(self.get_args_enum().STRIDES),
            padding=self.args.get(self.get_args_enum().PADDING).value(),
            dilation_rate=self.args.get(self.get_args_enum().DILATION_RATE),
            depth_multiplier=1,
            activation=self.args.get(self.get_args_enum().ACTIVATION).value(),
            depthwise_regularizer=gen_regularizer(
                self.args.get(self.get_args_enum().DEPTHWISE_REGULARIZER)
            ),
            pointwise_regularizer=gen_regularizer(
                self.args.get(self.get_args_enum().POINTWISE_REGULARIZER)
            ),
            depthwise_initializer=self.args.get(
                self.get_args_enum().DEPTHWISE_INITIALIZER
            ).value(),
            pointwise_initializer=self.args.get(
                self.get_args_enum().POINTWISE_INITIALIZER
            ).value(),
        )(input_tensor)
