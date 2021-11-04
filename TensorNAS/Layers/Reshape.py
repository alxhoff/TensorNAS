from enum import Enum, auto

from TensorNAS.Core.Layer import Layer
from TensorNAS.Core.Mutate import mutate_dimension


class Args(Enum):
    "Args needed for creating Reshape layer"
    TARGET_SHAPE = auto()


class Layer(Layer):
    def _gen_args(self, input_shape, target_shape):
        return {self.get_args_enum().TARGET_SHAPE: target_shape}

    def _mutate_target_shape(self):
        self.args[self.get_args_enum().TARGET_SHAPE] = mutate_dimension(
            self.args[self.get_args_enum().TARGET_SHAPE]
        )

    def get_output_shape(self):
        return self.args[self.get_args_enum().TARGET_SHAPE]

    def get_keras_layers(self, input_tensor):
        import tensorflow as tf

        return tf.keras.layers.Reshape(
            self.args.get(self.get_args_enum().TARGET_SHAPE),
            input_shape=self.inputshape.get(),
        )(input_tensor)
