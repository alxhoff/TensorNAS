from enum import Enum, auto

from TensorNAS.Core.BlockArchitecture import BlockArchitecture


class GhostNetArchitectureSubBlocks(Enum):
    GHOST_BLOCK = auto()
    CLASSIFICATION_BLOCK = auto()
    CONV2D = auto()


class Block(BlockArchitecture):
    MAX_SUB_BLOCKS = 3
    SUB_BLOCK_TYPES = GhostNetArchitectureSubBlocks

    def __init__(self, input_shape, class_count, batch_size, optimizer):
        self.class_count = class_count

        super().__init__(
            input_shape,
            parent_block=None,
            batch_size=batch_size,
            optimizer=optimizer,
        )

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Conv2D.Conv2D import Layer as Conv2D

        return [
            Conv2D(
                input_shape=input_shape,
                parent_block=self,
            )
        ]

    def generate_constrained_output_sub_blocks(self, input_shape):
        from TensorNAS.BlockTemplates.SubBlocks.TwoDClassificationBlock import (
            Block as TwoDClassificationBlock,
        )

        return [
            TwoDClassificationBlock(
                input_shape=input_shape,
                parent_block=self,
                class_count=self.class_count,
            )
        ]

    def generate_random_sub_block(self, input_shape, layer_type):
        from TensorNAS.BlockTemplates.SubBlocks.GhostBlock import Block as GhostBlock

        return [GhostBlock(input_shape=input_shape, parent_block=self)]
