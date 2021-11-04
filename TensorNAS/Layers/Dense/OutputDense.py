import TensorNAS.Core.Layer
from TensorNAS.Layers.Dense import Layer
from TensorNAS.Layers.Dense import Args as dense_args


class Layer(Layer):
    def _gen_args(self, input_shape, args):
        class_count = args.get(dense_args.UNITS)
        activation = TensorNAS.Core.Layer.ArgActivations.SOFTMAX

        if args:
            if self.get_args_enum().ACTIVATION in args:
                from TensorNAS.Core.Layer import ArgActivations

                activation = ArgActivations(args.get(self.get_args_enum().ACTIVATION))
            if self.get_args_enum().UNITS:
                class_count = args.get(self.get_args_enum().UNITS)

        if not args:
            raise Exception("Creating output dense layer without output class count")
        return {
            self.get_args_enum().UNITS: class_count,
            self.get_args_enum().ACTIVATION: activation,
        }
