from enum import Enum, auto

from tensornas.core.block import Block
from tensornas.core.layerblock import LayerBlock
from tensornas.layers import SupportedLayers
from tensornas.blocktemplates.subblocks.ExpansionBlock import ExpansionBlock

class SubBlockTypes(Enum):
    """
    Layers that can be used in the generation of a feature extraction block are enumerated here for random selection
    """
    NONE = auto()


class FeatureExtractionBlock(Block):
    """
    Layers that can be used in the extraction of features
    """

    MAX_SUB_BLOCKS = 2
    SUB_BLOCK_TYPES = SubBlockTypes

    def generate_random_sub_block(self, input_shape, layer_type):
        #TODO figure out how to do non sequential models
        """
        Fire blocks consist of a pointwise conv layer followed by any number of parallel separable 2d conv layers.
        This collection of separable conv layers is known as an expansion block.
        """
        return [
            LayerBlock(input_shape=input_shape, parent_block=self, layer_type=SupportedLayers.POINTWISECONV),
            ExpansionBlock(
                input_shape=input_shape, parent_block=self, layer_type=layer_type
            )
        ]
