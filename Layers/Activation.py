from TensorNAS.Core import EnumWithNone
from TensorNAS.Core.Layer import Layer
from TensorNAS.Core.LayerMutations import MutateActivation
from enum import auto


class Args(EnumWithNone):

    ACTIVATION = auto()


class Layer(Layer, MutateActivation):
    def _gen_args(self, input_shape, args):
        from TensorNAS.Core.Layer import gen_activation

        activation = gen_activation()

        if args:
            if self.get_args_enum().ACTIVATION in args:
                activation = args.get(self.get_args_enum().ACTIVATION)

        return {self.get_args_enum().ACTIVATION: activation}

    def get_output_shape(self):
        return self.inputshape.get()

    def get_keras_layers(self, input_tensor):
        import tensorflow as tf

        return tf.keras.layers.Activation(
            activation=self.args.get(self.get_args_enum().ACTIVATION).value()
        )(input_tensor)
