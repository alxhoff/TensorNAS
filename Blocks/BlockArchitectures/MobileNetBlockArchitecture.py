from TensorNAS.Core.BlockArchitecture import ClassificationBlockArchitecture
from enum import Enum, auto


class Block(ClassificationBlockArchitecture):

    MAX_SUB_BLOCKS = 3

    class SubBlocks(Enum):

        MOBILE_BLOCK = auto()
        CLASSIFICATION_BLOCK = auto()

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.Blocks.SubBlocks.MobilNetBlock import (
            Block as MobileNetBlock,
        )

        return [
            MobileNetBlock(
                input_shape=input_shape,
                parent_block=self,
            )
        ]

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

    def generate_random_sub_block(self, input_shape, layer_type):
        from TensorNAS.Blocks.SubBlocks.MobilNetBlock import (
            Block as MobileNetBlock,
        )

        return [MobileNetBlock(input_shape=input_shape, parent_block=self)]
