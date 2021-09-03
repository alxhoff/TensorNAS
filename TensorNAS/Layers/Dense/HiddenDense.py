import random

import TensorNAS.Core.LayerArgs as la
from TensorNAS.Core.Util import mutate_int, mutate_enum
from TensorNAS.Layers.Dense import Layer


class Layer(Layer):
    MAX_UNITS = 256

    def _gen_args(self, input_shape, args):
        class_count = random.randint(1, self.MAX_UNITS)
        activation = la.gen_activation()

        if args:
            if self.get_args_enum().ACTIVATION in args:
                from TensorNAS.Core.LayerArgs import ArgActivations

                activation = ArgActivations(args.get(self.get_args_enum().ACTIVATION))
            if self.get_args_enum().UNITS:
                class_count = args.get(self.get_args_enum().UNITS)

        return {
            self.get_args_enum().UNITS: class_count,
            self.get_args_enum().ACTIVATION: activation,
        }

    def _mutate_units(self):
        self.args[self.get_args_enum().UNITS] = mutate_int(
            self.args.get(self.get_args_enum().UNITS), 1, self.MAX_UNITS
        )

    def _mutate_activation(self):
        self.args[self.get_args_enum().ACTIVATION] = mutate_enum(
            self.args.get(self.get_args_enum().ACTIVATION), la.ArgActivations
        )
