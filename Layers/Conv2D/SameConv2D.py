import random

import TensorNAS.Core.Layer
from TensorNAS.Layers.Conv2D import Layer
from TensorNAS.Core.Mutate import MutationOperators
from TensorNAS.Core.LayerMutations import layer_mutation


class Layer(Layer):
    def _gen_args(self, input_shape, args):
        return {
            self.get_args_enum().FILTERS: random.randint(1, self.MAX_FILTER_COUNT),
            self.get_args_enum().KERNEL_SIZE: TensorNAS.Core.Layer.gen_2d_kernel_size(
                self.MAX_KERNEL_DIMENSION
            ),
            self.get_args_enum().STRIDES: (1, 1),
            self.get_args_enum().PADDING: TensorNAS.Core.Layer.ArgPadding.SAME,
            self.get_args_enum().DILATION_RATE: TensorNAS.Core.Layer.gen_2d_dilation(),
            self.get_args_enum().ACTIVATION: TensorNAS.Core.Layer.gen_activation(),
        }

    @layer_mutation
    def _mutate_strides(self, operator=MutationOperators.RANDOM):
        return "_mutate_strides", "Null mutation"

    @layer_mutation
    def _mutate_padding(self):
        return "_mutate_padding", "Null mutation"

    def get_keras_layers(self, input_tensor):
        import tensorflow as tf

        try:
            return tf.keras.layers.Conv2D(
                filters=self.args.get(self.get_args_enum().FILTERS),
                kernel_size=self.args.get(self.get_args_enum().KERNEL_SIZE),
                strides=self.args.get(self.get_args_enum().STRIDES),
                input_shape=self.inputshape.get(),
                activation=self.args.get(self.get_args_enum().ACTIVATION).value(),
                padding=self.args.get(self.get_args_enum().PADDING).value(),
                dilation_rate=self.args.get(self.get_args_enum().DILATION_RATE),
            )(input_tensor)
        except Exception as e:
            raise e
