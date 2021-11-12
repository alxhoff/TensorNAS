from TensorNAS.Core import EnumWithNone
from TensorNAS.Core.Layer import Layer


class Args(EnumWithNone):
    from enum import auto

    ACTIVATION = auto()


class Layer(Layer):
    def _gen_args(self, input_shape, args):
        from TensorNAS.Core.Layer import gen_activation

        activation = gen_activation()

        if args:
            if self.get_args_enum().ACTIVATION in args:
                activation = args.get(self.get_args_enum().ACTIVATION)

        return {self.get_args_enum().ACTIVATION: activation}

    def _mutate_activation(self):
        from TensorNAS.Core.Mutate import mutate_enum
        from TensorNAS.Core.Layer import ArgActivations

        self.args[self.get_args_enum().ACTIVATION] = mutate_enum(
            self.args[self.get_args_enum().ACTIVATION],
            ArgActivations,
        )

    def get_output_shape(self):
        return self.inputshape.get()

    def get_keras_layers(self, input_tensor):
        import tensorflow as tf

        return tf.keras.layers.Activation(
            activation=self.args.get(self.get_args_enum().ACTIVATION).value()
        )(input_tensor)
