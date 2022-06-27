import TensorNAS.Core.Layer
from TensorNAS.Layers.Conv2D import Layer
from TensorNAS.Core.Mutate import MutationOperators
from TensorNAS.Core.LayerMutations import layer_mutation


class Layer(Layer):
    def _gen_args(self, input_shape, args=None):
        from TensorNAS.Core.Layer import ArgActivations, ArgPadding

        filter_count = input_shape[-1]
        strides = (1, 1)
        dilation_rate = TensorNAS.Core.Layer.gen_2d_dilation()
        activation = ArgActivations.NONE
        padding = ArgPadding.SAME

        if args:
            if self.get_args_enum().FILTERS in args:
                filter_count = args.get(self.get_args_enum().FILTERS)
            if self.get_args_enum().PADDING in args:
                padding = ArgPadding(args.get(self.get_args_enum().PADDING))
            if self.get_args_enum().ACTIVATION in args:
                activation = ArgActivations(args.get(self.get_args_enum().ACTIVATION))
            if self.get_args_enum().DILATION_RATE in args:
                dilation_rate = args.get(self.get_args_enum().DILATION_RATE)
            if self.get_args_enum().STRIDES in args:
                strides = args.get(self.get_args_enum().STRIDES)

        return {
            self.get_args_enum().FILTERS: filter_count,
            self.get_args_enum().KERNEL_SIZE: (1, 1),
            self.get_args_enum().PADDING: padding,
            self.get_args_enum().STRIDES: strides,
            self.get_args_enum().DILATION_RATE: dilation_rate,
            self.get_args_enum().ACTIVATION: activation,
        }

    @layer_mutation
    def _mutate_kernel_size(self, operator=MutationOperators.RANDOM):
        """
        A pointwise conv requires a 1x1 kernel, thus we cannot mutate it. This function is here to override the
        standard conv mutations found in the super class.
        """
        return "_mutate_kernel_size", "Null mutation"

    def get_keras_layers(self, input_tensor):
        import tensorflow as tf

        return tf.keras.layers.Conv2D(
            filters=self.args.get(self.get_args_enum().FILTERS),
            kernel_size=self.args.get(self.get_args_enum().KERNEL_SIZE),
            strides=self.args.get(self.get_args_enum().STRIDES),
            input_shape=self.inputshape.get(),
            activation=self.args.get(self.get_args_enum().ACTIVATION).value(),
            padding=self.args.get(self.get_args_enum().PADDING).value(),
            dilation_rate=self.args.get(self.get_args_enum().DILATION_RATE),
        )(input_tensor)
