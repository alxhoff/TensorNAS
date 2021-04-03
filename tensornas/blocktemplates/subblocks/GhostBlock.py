from enum import Enum, auto

from tensornas.core.block import Block
from tensornas.core.layerblock import LayerBlock
from tensornas.layers import SupportedLayers


class SubBlockTypes(Enum):
    """
    Layers that can be used in the generation of a feature extraction block are enumerated here for random selection
    """

    POINTWISE_CONV = auto()
    DEPTHWISE_CONV = auto()


class GhostBlock(Block):

    MAX_SUB_BLOCKS = 2
    SUB_BLOCK_TYPES = SubBlockTypes

    """pass out_shape as input to Ghost Block and also ratio number for no. of channels to be processed by pointwise conv and remaining by depthwise conv.
 
    DepthwiseConv has a parameter multiplier for number of output channels to be generated for each input channel
    """

    def generate_random_sub_block(self, input_shape, layer_type):
        pwconv_block = LayerBlock(
            input_shape=input_shape,
            parent_block=self,
            layer_type=SupportedLayers.POINTWISECONV2D,
        )
        dwconv_block = LayerBlock(
            input_shape=input_shape,
            parent_block=self,
            layer_type=SupportedLayers.DEPTHWISECONV2D,
        )
        return [pwconv_block, dwconv_block]
