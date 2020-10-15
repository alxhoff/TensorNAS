from tensornas.layers.Dense import Layer
from tensornas.core.layerargs import ArgActivations


class Layer(Layer):
    def _gen_args(self, input_shape, args):
        if not args:
            raise Exception("Creating output dense layer without output class count")
        return {
            self.get_args_enum().UNITS.value: args,
            self.get_args_enum().ACTIVATION.value: ArgActivations.SOFTMAX.value,
        }
