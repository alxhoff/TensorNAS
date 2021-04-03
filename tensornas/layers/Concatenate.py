from enum import Enum, auto

import tensorflow as tf

from tensornas.core.layer import NetworkLayer


class Args(Enum):
    AXIS = auto()
    LAYERS = auto()


class Layer(NetworkLayer):
    def _gen_args(self, input_shape, args):
        axis = -1
        layers = None
        if self.get_args_enum().AXIS in args:
            axis = args.get(self.get_args_enum().AXIS)
        if self.get_args_enum().LAYERS in args:
            layers = args.get(self.get_args_enum().AXIS)
        return {self.get_args_enum().AXIS: axis, self.get_args_enum().LAYERS: layers}

    def get_output_shape(self):
        layers = self.args.get(self.get_args_enum().LAYERS)
        axis = self.args.get(self.get_args_enum().AXIS)
        assert len(layers) > 0
        shape_len = len(layers[0])
        dims = [0] * shape_len
        output_shapes = [layer.get_output_shape() for layer in layers]
        for i, os in enumerate(output_shapes):
            if i == axis:
                continue
            for dim in dims:
                dims[dim] = max(dims[dim], os[dim])

        for os in output_shapes:
            dim[axis] += os[axis]

        return dim

    def get_keras_layer(self, input_tensor):
        # TODO
        keras_layers = [
            layer.get_keras_layer()
            for layer in self.args.get(self.get_args_enum().LAYERS)
        ]
        return [
            tf.keras.layers.Concatenate(
                keras_layers, axis=self.args.get(self.get_args_enum().AXIS)
            )
        ]
