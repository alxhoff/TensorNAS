from TensorNAS.Core.Block import Block
from enum import Enum, auto


class Block(Block):
    """
    And Expand block is a block used in SqueezeNet following a squeeze block within a Fire Module. A Expand block is a
    collection of parallel convolutional Layers.
    """

    MAX_SUB_BLOCKS = 5
    MIN_SUB_BLOCKS = 2

    class SubBlocks(Enum):

        CONV2D = auto()

    def generate_sub_block(self, input_shape, layer_type):
        """
        Care must be taken here that the input shape is the input to the Expand block as all Layers are in parallel
        and thus take the same input, ie. the input to the block.
        """
        from TensorNAS.Layers.Conv2D.SameConv2D import Layer as SameConv2D

        if layer_type == self.SubBlocks.CONV2D:
            return [
                SameConv2D(
                    input_shape=self.input_shape,
                    parent_block=self,
                )
            ]
        return []

    def get_keras_layers(self, input_tensor):
        """
        As an expand block creates a number of parallel 2D Conv Layers the functional Tensorflow API must be used.
        The use of a concatenation operation allows for the Layers to be parallelized, the one requirement of using
        the functional API is that the parallelized Layers must be constructed with knowledge of the input layer
        from which the input should be input in parallel. If this block is being called from a Fire Block this would
        mean that the squeeze layer must be passed in.
        """
        import tensorflow as tf

        layers = [sb.get_keras_layers(input_tensor) for sb in self.middle_blocks]
        if len(layers) > 1:
            return tf.keras.layers.Concatenate()(layers)
        else:
            return layers[0]
