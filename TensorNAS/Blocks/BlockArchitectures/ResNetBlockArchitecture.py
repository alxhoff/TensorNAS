from TensorNAS.Core.BlockArchitecture import ClassificationBlockArchitecture


class Block(ClassificationBlockArchitecture):
    from enum import Enum

    MAX_SUB_BLOCKS = 1

    class SubBlocks(Enum):
        from enum import auto

        RESIDUAL_BLOCK = auto()
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

    def generate_random_sub_block(self, input_shape, layer_type):
        from TensorNAS.Blocks.SubBlocks.ResidualBlock import (
            Block as ResidualBlock,
        )

        return [ResidualBlock(input_shape=input_shape, parent_block=self)]
