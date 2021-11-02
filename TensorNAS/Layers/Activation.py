from enum import Enum, auto
from TensorNAS.Core.Layer import NetworkLayer
import TensorNAS.Core.LayerArgs as la
from TensorNAS.Core.Util import mutate_enum


class Args(Enum):

    ACTIVATION = auto()


class Layer(NetworkLayer):
    def _gen_args(self, input_shape, args):

        activation = la.gen_activation()

        if args:
            if self.get_args_enum().ACTIVATION in args:
                activation = args.get(self.get_args_enum().ACTIVATION)

        return {self.get_args_enum().ACTIVATION: activation}

    def _mutate_activation(self):
        self.args[self.get_args_enum().ACTIVATION] = mutate_enum(
            self.args[self.get_args_enum().ACTIVATION], la.ArgActivations
        )

    def get_output_shape(self):
        return self.inputshape.get()

    def get_keras_layer(self, input_tensor):
        import tensorflow as tf

        return tf.keras.layers.Activation(
            activation=self.args.get(self.get_args_enum().ACTIVATION).value
        )(input_tensor)
