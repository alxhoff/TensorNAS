from TensorNAS.Core import EnumWithNone
from TensorNAS.Core.Layer import Layer


class Args(EnumWithNone):
    from enum import auto

    RATE = auto()


class Layer(Layer):
    MAX_RATE = 0.5

    def _gen_args(self, input_shape, args):
        from TensorNAS.Core.Layer import gen_dropout

        max = gen_dropout(self.MAX_RATE)

        if args:
            if self.get_args_enum().RATE in args:
                max = args.get(self.get_args_enum().RATE)

        return {self.get_args_enum().RATE: max}

    def _mutate_rate(self):
        from TensorNAS.Core.Mutate import mutate_unit_interval

        self.args[self.get_args_enum().RATE] = mutate_unit_interval(
            self.args[self.get_args_enum().RATE], 0, self.MAX_RATE
        )

    def get_output_shape(self):
        return self.inputshape.get()

    def get_keras_layers(self, input_tensor):
        import tensorflow as tf

        return tf.keras.layers.Dropout(
            rate=self.args.get(self.get_args_enum().RATE),
            input_shape=self.inputshape.get(),
        )(input_tensor)
