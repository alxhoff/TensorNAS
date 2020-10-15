from enum import Enum, auto

from tensornas.core.networklayer import NetworkLayer
from tensornas.core.util import dimension_mag, mutate_dimension


def gen_Reshape_args(input_shape, target_shape):
    return {ReshapeArgs.TARGET_SHAPE.value: target_shape}


class ReshapeArgs(Enum):
    "Args needed for creating Reshape layer"
    TARGET_SHAPE = auto()


class ReshapeLayer(NetworkLayer):
    MUTATABLE_PARAMETERS = 0

    def _target_shape(self):
        return self.args.get(ReshapeArgs.TARGET_SHAPE.value, self.inputshape.get())

    def _mutate_target_shape(self):
        self.args[ReshapeArgs.TARGET_SHAPE.value] = mutate_dimension(
            self._target_shape()
        )

    def repair(self):
        self.inputshape.set(self.outputshape.get())
        self._mutate_target_shape()

    def mutate(self):
        self._mutate_target_shape()

    def validate(self, repair=True):
        input_mag = dimension_mag(list(self.inputshape.get()))
        output_mag = dimension_mag(list(self.get_output_shape()))

        if not input_mag == output_mag:
            if repair:
                while not input_mag == output_mag:
                    self.repair()
                    input_mag = dimension_mag(list(self.inputshape.get()))
                    output_mag = dimension_mag(list(self.get_output_shape()))
            else:
                return False
        return True

    def get_output_shape(self):
        return self._target_shape()

    def get_keras_layer(self):
        target_shape = self.args.get(ReshapeArgs.TARGET_SHAPE.value)
        return tf.keras.layers.Reshape(target_shape)
