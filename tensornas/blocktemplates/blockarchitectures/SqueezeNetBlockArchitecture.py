from enum import Enum, auto

from tensornas.blocktemplates.subblocks.TwoDClassificationBlock import (
    TwoDClassificationBlock,
)
from tensornas.blocktemplates.subblocks.FireBlock import FireBlock
from tensornas.core.blockarchitecture import BlockArchitecture


class SqueezeNetArchitectureSubBlocks(Enum):
    FIRE_BLOCK = auto()
    """classification block some of the layers like flatten is added to the input constrained block and  
     output dense is already added to the constrained output block"""


class SqueezeNetBlockArchitecture(BlockArchitecture):
    MAX_SUB_BLOCKS = 3
    SUB_BLOCK_TYPES = SqueezeNetArchitectureSubBlocks

    def __init__(self, input_shape, class_count):
        self.class_count = class_count

        super().__init__(input_shape, parent_block=None, layer_type=None)

    def generate_constrained_output_sub_blocks(self, input_shape):
        return [
            TwoDClassificationBlock(
                input_shape=input_shape,
                parent_block=self,
                class_count=self.class_count,
            )
        ]

    def generate_random_sub_block(self, input_shape, layer_type):
        if layer_type == self.SUB_BLOCK_TYPES.FIRE_BLOCK:
            return [
                FireBlock(
                    input_shape=input_shape, parent_block=self, layer_type=layer_type
                )
            ]
        return []
