from enum import Enum, auto
from tensorflow import keras

from tensornas.core.block import Block
from tensornas.core.layerblock import LayerBlock
from tensornas.layers import SupportedLayers


class SubBlockTypes(Enum):
    CONV2D = auto()


class ExpandBlock(Block):
    """
    And Expand block is a block used in SqueezeNet following a squeeze block within a Fire Module. A Expand block is a
    collection of parallel convolutional layers.
    """

    MAX_SUB_BLOCKS = 5
    MIN_SUB_BLOCK = 2
    SUB_BLOCK_TYPES = SubBlockTypes

    def generate_random_sub_block(self, input_shape, layer_type):
        """
        Care must be taken here that the input shape is the input to the Expand block as all layers are in parallel
        and thus take the same input, ie. the input to the block.
        """
        if layer_type == self.SUB_BLOCK_TYPES.CONV2D:
            return [
                LayerBlock(
                    input_shape=self.input_shape,
                    parent_block=self,
                    layer_type=SupportedLayers.SAMECONV2D,
                )
            ]
        return []

    def get_keras_layers(self, input_tensor):
        """
        As an expand block creates a number of parallel 2D Conv layers the functional Tensorflow API must be used.
        The use of a concatenation operation allows for the layers to be parallelized, the one requirement of using
        the functional API is that the parallelized layers must be constructed with knowledge of the input layer
        from which the input should be input in parallel. If this block is being called from a Fire Block this would
        mean that the squeeze layer must be passed in.
        """
        layers = [sb.get_keras_layers(input_tensor) for sb in self.middle_blocks]
        if len(layers) > 1:
            return keras.layers.Concatenate()(layers)
        else:
            return layers[0]
