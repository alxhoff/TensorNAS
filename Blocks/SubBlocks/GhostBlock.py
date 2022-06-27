from TensorNAS.Core.Block import Block
from enum import Enum, auto


class Block(Block):

    MAX_SUB_BLOCKS = 2

    class SubBlocks(Enum):

        GHOST_BLOCK = auto()

    """pass out_shape as input to Ghost Block and also ratio number for no. of channels to be processed by pointwise conv and remaining by depthwise conv.
 
    DepthwiseConv has a parameter multiplier for number of output channels to be generated for each input channel
    """

    def generate_sub_block(self, input_shape, layer_type):
        from TensorNAS.Layers.Conv2D.PointwiseConv2D import Layer as PointwiseConv2D
        from TensorNAS.Layers.Conv2D.DepthwiseConv2D import Layer as DepthwiseConv2D

        if layer_type == self.SubBlocks.GHOST_BLOCK:
            pwconv_block = PointwiseConv2D(
                input_shape=input_shape,
                parent_block=self,
            )
            dwconv_block = DepthwiseConv2D(
                input_shape=input_shape,
                parent_block=self,
            )
            return [pwconv_block, dwconv_block]
        return []
