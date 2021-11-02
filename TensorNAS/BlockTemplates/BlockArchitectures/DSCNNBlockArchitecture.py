from enum import Enum, auto

from TensorNAS.BlockTemplates.SubBlocks.DSCNN.DSCNNSeperableDepthwiseConv2DBlock import (
    Block as DSCNNMidBlock,
)
from TensorNAS.BlockTemplates.SubBlocks.DSCNN.DSCNNOutputBlock import (
    Block as DSCNNOutputBlock,
)
from TensorNAS.BlockTemplates.SubBlocks.DSCNN.DSCNNInputBlock import (
    Block as DSCNNInputBlock,
)

from TensorNAS.Core.BlockArchitecture import ClassificationBlockArchitecture


class DSCNNSubBlocks(Enum):
    DSCNN_INPUT_BLOCK = auto()
    DSCNN_SEPERABLE_DEPTHWISE_CONV2D_BLOCK = auto()
    DSCNN_OUTPUT_BLOCK = auto()


class Block(ClassificationBlockArchitecture):

    MIN_SUB_BLOCKS = 4
    MAX_SUB_BLOCKS = 4
    SUB_BLOCK_TYPES = DSCNNSubBlocks

    def generate_constrained_input_sub_blocks(self, input_shape):
        return [
            DSCNNInputBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=self.SUB_BLOCK_TYPES.DSCNN_INPUT_BLOCK,
            )
        ]

    def generate_constrained_output_sub_blocks(self, input_shape):
        return [
            DSCNNOutputBlock(
                input_shape=input_shape, parent_block=self, class_count=self.class_count
            )
        ]

    def generate_random_sub_block(self, input_shape, layer_type):
        return [
            DSCNNMidBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=self.SUB_BLOCK_TYPES.DSCNN_SEPERABLE_DEPTHWISE_CONV2D_BLOCK,
            )
        ]
