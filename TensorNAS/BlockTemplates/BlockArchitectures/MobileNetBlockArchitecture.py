from enum import Enum, auto

from TensorNAS.BlockTemplates.SubBlocks.TwoDClassificationBlock import (
    Block as TwoDClassificationBlock,
)
from TensorNAS.BlockTemplates.SubBlocks.MobilNetBlock import Block as MobileNetBlock
from TensorNAS.Core.BlockArchitecture import BlockArchitecture


class MobileNetArchitectureSubBlocks(Enum):
    MOBILE_BLOCK = auto()
    CLASSIFICATION_BLOCK = auto()


class Block(BlockArchitecture):
    MAX_SUB_BLOCKS = 3
    SUB_BLOCK_TYPES = MobileNetArchitectureSubBlocks

    def __init__(self, input_shape, class_count, batch_size, optimizer):
        self.class_count = class_count

        super().__init__(
            input_shape,
            parent_block=None,
            layer_type=None,
            batch_size=batch_size,
            optimizer=optimizer,
        )

    def generate_constrained_input_sub_blocks(self, input_shape):
        return [
            MobileNetBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=self.SUB_BLOCK_TYPES.MOBILE_BLOCK,
            )
        ]

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
            MobileNetBlock(
                input_shape=input_shape, parent_block=self, layer_type=layer_type
            )
        ]
