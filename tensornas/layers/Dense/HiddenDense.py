import random

import tensornas.core.layerargs as la
from tensornas.core.util import mutate_int, mutate_enum
from tensornas.layers.Dense import Layer


class Layer(Layer):
    MAX_UNITS = 256

    def _gen_args(self, input_shape, args):
        return {
            self.get_args_enum().UNITS: random.randint(1, self.MAX_UNITS),
            self.get_args_enum().ACTIVATION: la.gen_activation(),
        }

    def _mutate_units(self):
        self.args[self.get_args_enum().UNITS] = mutate_int(
            self.args.get(self.get_args_enum().UNITS), 1, self.MAX_UNITS
        )

    def _mutate_activation(self):
        self.args[self.get_args_enum().ACTIVATION] = mutate_enum(
            self.args.get(self.get_args_enum().ACTIVATION), la.ArgActivations
        )
