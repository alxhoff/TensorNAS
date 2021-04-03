from enum import Enum, auto

from tensornas.blocktemplates.subblocks.TwoDClassificationBlock import (
    TwoDClassificationBlock,
)
from tensornas.blocktemplates.subblocks.ResidualBlock import ResidualBlock
from tensornas.core.blockarchitecture import BlockArchitecture


class ResNetArchitectureSubBlocks(Enum):
    RESIDUAL_BLOCK = auto()
    CLASSIFICATION_BLOCK = auto()


class ResNetBlockArchitecture(BlockArchitecture):
    MAX_SUB_BLOCKS = 5
    SUB_BLOCK_TYPES = ResNetArchitectureSubBlocks

    def __init__(self, input_shape, class_count):
        self.class_count = class_count

        super().__init__(input_shape, parent_block=None, layer_type=None)

    def generate_constrained_output_sub_blocks(self, input_shape):
        return [
            TwoDClassificationBlock(
                input_shape=input_shape,
                parent_block=self,
                class_count=self.class_count,
                layer_type=self.SUB_BLOCK_TYPES.CLASSIFICATION_BLOCK,
            )
        ]

    def generate_random_sub_block(self, input_shape, layer_type):
        return [
            ResidualBlock(
                input_shape=input_shape, parent_block=self, layer_type=layer_type
            )
        ]
