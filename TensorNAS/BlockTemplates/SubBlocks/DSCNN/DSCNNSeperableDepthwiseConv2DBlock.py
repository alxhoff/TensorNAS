from enum import Enum, auto

from TensorNAS.Core.Block import Block


class DSCNNSubBlocks(Enum):
    DSCNN_CONV_BLOCK = auto()
    DSCNN_DEPTHWISE_CONV_BLOCK = auto()


class Block(Block):

    MAX_SUB_BLOCKS = 1
    SUB_BLOCK_TYPES = DSCNNSubBlocks

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.BlockTemplates.SubBlocks.DSCNN.DSCNNConv2DBlock import (
            Block as Conv2DBlock,
        )
        from TensorNAS.BlockTemplates.SubBlocks.DSCNN.DSCNNDepthwiseConv2DBlock import (
            Block as DWConv2DBlock,
        )

        blocks = []

        blocks.append(
            DWConv2DBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=DSCNNSubBlocks.DSCNN_DEPTHWISE_CONV_BLOCK,
            )
        )
        blocks.append(
            Conv2DBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=DSCNNSubBlocks.DSCNN_CONV_BLOCK,
            )
        )

        return blocks
