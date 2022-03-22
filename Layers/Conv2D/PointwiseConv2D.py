import TensorNAS.Core.Layer
from TensorNAS.Layers.Conv2D import Layer
from TensorNAS.Core.Mutate import MutationOperators


class Layer(Layer):
    def _gen_args(self, input_shape, args=None):
        filter_count = input_shape[-1]

        return {
            self.get_args_enum().FILTERS: filter_count,
            self.get_args_enum().KERNEL_SIZE: (1, 1),
            self.get_args_enum().STRIDES: (1, 1),
            self.get_args_enum().PADDING: TensorNAS.Core.Layer.gen_padding(),
            self.get_args_enum().DILATION_RATE: TensorNAS.Core.Layer.gen_2d_dilation(),
            self.get_args_enum().ACTIVATION: TensorNAS.Core.Layer.gen_activation(),
        }

    def _mutate_kernel_size(self, operator=MutationOperators.SYNC_STEP):
        """
        A pointwise conv requires a 1x1 kernel, thus we cannot mutate it. This function is here to override the
        standard conv mutations found in the super class.
        """
        return "Null mutation"

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
