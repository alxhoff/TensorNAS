# class to create Expansion block layers as objects and add constraints on input
# fixed layers and output fixed layers for the Expansion Block

from enum import Enum, auto

from tensornas.core.block import Block
from tensornas.core.layerblock import LayerBlock
from tensornas.layers import SupportedLayers

"""
An expansion block is some number of seperable 2d convolution layers in parallel, usually found after a pointwise 2d
convolution as part of a fire block. Fire blocks are found in SqueezeNet architectures.
"""


class SubBlockTypes(Enum):
    """
    Layers that can be used in the generation of a feature extraction block are enumerated here for random selection
    """
    SEPERABLECONV2D = auto()


class ExpansionBlock(Block):
    """
    Layers that can be used in the extraction of features
    """

    MAX_SUB_BLOCKS = 3
    SUB_BLOCK_TYPES = SubBlockTypes

    def generate_random_sub_block(self, input_shape, layer_type):
        return [
            LayerBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=SupportedLayers.SEPARABLECONV2D,
            )
        ]
