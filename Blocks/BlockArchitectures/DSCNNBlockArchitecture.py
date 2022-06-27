from TensorNAS.Core.BlockArchitecture import ClassificationBlockArchitecture
from enum import Enum, auto


class Block(ClassificationBlockArchitecture):

    MIN_SUB_BLOCKS = 0
    MAX_SUB_BLOCKS = 0

    class SubBlocks(Enum):

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

    def generate_constrained_middle_sub_blocks(self, input_shape, args=None):
        from TensorNAS.Blocks.SubBlocks.DSCNN.DSCNNMidBlock import (
            Block as DSCNNMidBlock,
        )

        layers = []

        layers.append(DSCNNMidBlock(input_shape=input_shape, parent_block=self))

        for i in range(3):
            layers.append(
                DSCNNMidBlock(
                    input_shape=layers[-1].get_output_shape(), parent_block=self
                )
            )

        return layers

    def generate_sub_block(self, input_shape, layer_type):
        from TensorNAS.Blocks.SubBlocks.DSCNN.DSCNNMidBlock import (
            Block as DSCNNMidBlock,
        )

        if layer_type == self.SubBlocks.DSCNN_SEPERABLE_DEPTHWISE_CONV2D_BLOCK:
            return [
                DSCNNMidBlock(
                    input_shape=input_shape,
                    parent_block=self,
                )
            ]

        return []
