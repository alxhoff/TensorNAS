from TensorNAS.Core.Block import Block


class Block(Block):

    MAX_SUB_BLOCKS = 1

    from enum import Enum

    class SubBlocks(Enum):
        from enum import auto

        DSCNN_CONV_BLOCK = auto()
        DSCNN_DEPTHWISE_CONV_BLOCK = auto()

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.Blocks.SubBlocks.DSCNN.DSCNNConv2DBlock import (
            Block as Conv2DBlock,
        )
        from TensorNAS.Blocks.SubBlocks.DSCNN.DSCNNDepthwiseConv2DBlock import (
            Block as DWConv2DBlock,
        )

        blocks = []

        blocks.append(
            DWConv2DBlock(
                input_shape=input_shape,
                parent_block=self,
            )
        )
        blocks.append(
            Conv2DBlock(
                input_shape=input_shape,
                parent_block=self,
            )
        )

        return blocks
