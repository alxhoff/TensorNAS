from TensorNAS.Core.BlockArchitecture import ClassificationBlockArchitecture
from enum import Enum, auto


class Block(ClassificationBlockArchitecture):

    MAX_SUB_BLOCKS = 1

    class SubBlocks(Enum):

        SHUFFLENET_BLOCK = auto()
        CLASSIFICATION_BLOCK = auto()

    def generate_constrained_output_sub_blocks(self, input_shape):
        from TensorNAS.Blocks.SubBlocks.TwoDClassificationBlock import (
            Block as TwoDClassificationBlock,
        )

        return [
            TwoDClassificationBlock(
                input_shape=input_shape,
                parent_block=self,
                class_count=self.class_count,
            )
        ]

    def generate_sub_block(self, input_shape, layer_type):
        from TensorNAS.Blocks.SubBlocks.ShuffleNetBlock import (
            Block as ShuffleNetBlock,
        )

        if layer_type == self.SubBlocks.SHUFFLENET_BLOCK:
            return [ShuffleNetBlock(input_shape=input_shape, parent_block=self)]
        return []
