from TensorNAS.Core.Block import Block
from enum import Enum, auto


class Block(Block):
    """
    Layers that can be used in the extraction of features
    """

    MAX_SUB_BLOCKS = 2

    class SubBlocks(Enum):

        NONE = auto()

    def generate_sub_block(self, input_shape, layer_type):
        """
        Fire blocks consist of a pointwise conv layer followed by any number of parallel separable 2d conv Layers.
        This collection of separable conv Layers is known as an expansion block.
        """
        from TensorNAS.Layers.Conv2D.PointwiseConv2D import Layer as PointwiseConv2D
        from TensorNAS.Blocks.SubBlocks.ExpandBlock import Block as ExpandBlock

        pwconv_block = PointwiseConv2D(
            input_shape=input_shape,
            parent_block=self,
        )
        expand_block = ExpandBlock(
            input_shape=pwconv_block.get_output_shape(),
            parent_block=self,
        )
        return [pwconv_block, expand_block]

    def get_keras_layers(self, input_tensor):
        """
        As the fire block generates parallel Layers within its expand block the functional Tensorflow API must be used.
        The expand block must be fed the squeeze layer in such that the functional API can correctly construct the
        parallelized Layers. Thus the get_keras_layers function must be overwritten to handle this passing.
        """
        squeeze_layer = self.middle_blocks[0].get_keras_layers(input_tensor)
        expand_layer = self.middle_blocks[1].get_keras_layers(squeeze_layer)
        return expand_layer
