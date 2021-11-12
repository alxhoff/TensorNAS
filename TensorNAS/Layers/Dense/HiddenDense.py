from TensorNAS.Layers.Dense import Layer


class Layer(Layer):
    MAX_UNITS = 256

    def _mutate_units(self):
        from TensorNAS.Core.Mutate import mutate_int

        self.args[self.get_args_enum().UNITS] = mutate_int(
            self.args.get(self.get_args_enum().UNITS), 1, self.MAX_UNITS
        )

    def _mutate_activation(self):
        from TensorNAS.Core.Layer import ArgActivations
        from TensorNAS.Core.Mutate import mutate_enum

        self.args[self.get_args_enum().ACTIVATION] = mutate_enum(
            self.args.get(self.get_args_enum().ACTIVATION),
            ArgActivations,
        )
