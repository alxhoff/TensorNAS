import tensornas.core.layerargs as la
from tensornas.layers.Dense import Layer


class Layer(Layer):
    def _gen_args(cls, input_shape, args):
        if not args:
            raise Exception("Creating output dense layer without output class count")
        return {
            cls.get_args_enum().UNITS: args,
            cls.get_args_enum().ACTIVATION: la.ArgActivations.SOFTMAX,
        }
