from tensornas.core.block import Block

from tensornas.layers import Layers


class LayerBlock(Block):
    """
    Layer blocks represent the lowest level blocks of the block architecture, they exist to house a single
    keras layer with which they interface to provide the appropriate information to the higher level
    blocks, eg. input/output shapes.

    To manage the keras layer the block contains a ModelLayer object which works to manage the properties of the
    tensorflow/keras layer which can then be generated when required.
    """

    MAX_SUB_BLOCKS = 0
    SUB_BLOCK_TYPES = None

    def __init__(self, input_shape, parent_block, layer_type, args=None):
        if not input_shape:
            input_shape = parent_block._get_cur_output_shape()
        layer = eval("Layers." + layer_type.name + ".value.Layer")
        self.layer = layer(input_shape=input_shape, args=args)

        super().__init__(
            input_shape=input_shape, parent_block=parent_block, layer_type=layer_type
        )

    def mutate(self, verbose=False):
        return self.layer.mutate(verbose)

    def get_output_shape(self):
        return self.layer.get_output_shape()

    def get_keras_layers(self, input_tensor):
        return self.layer.get_keras_layer(input_tensor)

    def print_self(self):
        self.layer.print()

    def generate_random_sub_block(self, layer_type):
        pass

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape
        self.layer.inputshape.set(input_shape)
