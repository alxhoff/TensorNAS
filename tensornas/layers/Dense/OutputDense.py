import tensornas.core.layerargs as la
from tensornas.layers.Dense import Layer
from tensornas.layers.Dense import Args as dense_args


class Layer(Layer):
    def _gen_args(self, input_shape, args):
        class_count = args.get(dense_args.UNITS)
        if not args:
            raise Exception("Creating output dense layer without output class count")
        return {
            self.get_args_enum().UNITS: class_count,
            self.get_args_enum().ACTIVATION: la.ArgActivations.SOFTMAX,
        }
