from TensorNAS.Core.BlockArchitecture import ClassificationBlockArchitecture


class Block(ClassificationBlockArchitecture):
    from enum import Enum

    MAX_SUB_BLOCKS = 1

    class SubBlocks(Enum):
        from enum import auto

        EFFNET_BLOCK = auto()
        CLASSIFICATION_BLOCK = auto()

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.Blocks.SubBlocks.EffNetBlock import Block as EffNetBlock

        return [
            EffNetBlock(
                input_shape=input_shape,
                parent_block=self,
            )
        ]

    def generate_constrained_output_sub_blocks(self, input_shape):
        from TensorNAS.Blocks.SubBlocks.TwoDClassificationBlock import (
            Block as TwoDClassificationBlock,
        )

        layers = []
        # Layers.append(GlobalAveragePool2D(input_shape=input_shape))
        layers.append(
            TwoDClassificationBlock(
                input_shape=input_shape,
                parent_block=self,
                class_count=self.class_count,
            )
        )
        return layers

    def generate_random_sub_block(self, input_shape, layer_type):
        from TensorNAS.Blocks.SubBlocks.EffNetBlock import Block as EffNetBlock

        return [EffNetBlock(input_shape=input_shape, parent_block=self)]
