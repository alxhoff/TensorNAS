from enum import Enum, auto

from TensorNAS.BlockTemplates.SubBlocks.TwoDClassificationBlock import (
    Block as TwoDClassificationBlock,
)
from TensorNAS.BlockTemplates.SubBlocks.ShuffleNetBlock import Block as ShuffleNetBlock
from TensorNAS.Core.BlockArchitecture import BlockArchitecture


class ShuffleNetArchitectureSubBlocks(Enum):
    SHUFFLENET_BLOCK = auto()
    CLASSIFICATION_BLOCK = auto()


class Block(BlockArchitecture):
    MAX_SUB_BLOCKS = 1
    SUB_BLOCK_TYPES = ShuffleNetArchitectureSubBlocks

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
        if layer_type == self.SUB_BLOCK_TYPES.SHUFFLENET_BLOCK:
            return [
                ShuffleNetBlock(
                    input_shape=input_shape, parent_block=self, layer_type=layer_type
                )
            ]
        return []
