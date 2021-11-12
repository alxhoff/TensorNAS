from TensorNAS.Core.BlockArchitecture import ClassificationBlockArchitecture


class Block(ClassificationBlockArchitecture):
    from enum import Enum

    MIN_SUB_BLOCKS = 4
    MAX_SUB_BLOCKS = 4

    class SubBlocks(Enum):
        from enum import auto

        DSCNN_INPUT_BLOCK = auto()
        DSCNN_SEPERABLE_DEPTHWISE_CONV2D_BLOCK = auto()
        DSCNN_OUTPUT_BLOCK = auto()

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.Blocks.SubBlocks.DSCNN.DSCNNInputBlock import (
            Block as DSCNNInputBlock,
        )

        return [
            DSCNNInputBlock(
                input_shape=input_shape,
                parent_block=self,
            )
        ]

    def generate_constrained_output_sub_blocks(self, input_shape):
        from TensorNAS.Blocks.SubBlocks.DSCNN.DSCNNOutputBlock import (
            Block as DSCNNOutputBlock,
        )

        return [
            DSCNNOutputBlock(
                input_shape=input_shape, parent_block=self, class_count=self.class_count
            )
        ]

    def generate_random_sub_block(self, input_shape, layer_type):
        from TensorNAS.Blocks.SubBlocks.DSCNN.DSCNNSeperableDepthwiseConv2DBlock import (
            Block as DSCNNMidBlock,
        )

        return [
            DSCNNMidBlock(
                input_shape=input_shape,
                parent_block=self,
            )
        ]
