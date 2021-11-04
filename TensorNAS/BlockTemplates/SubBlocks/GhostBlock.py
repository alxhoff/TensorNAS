from enum import Enum, auto

from TensorNAS.Core.Block import Block


class SubBlockTypes(Enum):
    """
    Layers that can be used in the generation of a feature extraction block are enumerated here for random selection
    """

    POINTWISE_CONV2D = auto()
    DEPTHWISE_CONV2D = auto()


class Block(Block):

    MAX_SUB_BLOCKS = 2
    SUB_BLOCK_TYPES = SubBlockTypes

    """pass out_shape as input to Ghost Block and also ratio number for no. of channels to be processed by pointwise conv and remaining by depthwise conv.
 
    DepthwiseConv has a parameter multiplier for number of output channels to be generated for each input channel
    """

    def generate_random_sub_block(self, input_shape, layer_type):
        from TensorNAS.Layers.Conv2D.PointwiseConv2D import Layer as PointwiseConv2D
        from TensorNAS.Layers.Conv2D.DepthwiseConv2D import Layer as DepthwiseConv2D

        pwconv_block = PointwiseConv2D(
            input_shape=input_shape,
            parent_block=self,
        )
        dwconv_block = DepthwiseConv2D(
            input_shape=input_shape,
            parent_block=self,
        )
        return [pwconv_block, dwconv_block]
