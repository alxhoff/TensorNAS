from TensorNAS.Core.Block import Block
from enum import Enum, auto


class Block(Block):

    MIN_SUB_BLOCKS = 3
    MAX_SUB_BLOCKS = 3

    class SubBlocks(Enum):

        RESNET_MIDDLE_BLOCK = auto()

    def generate_constrained_middle_sub_blocks(self, input_shape):
        from TensorNAS.Blocks.SubBlocks.ResNet.ResNetMiddleBlock import (
            Block as MidBlock,
        )
        from TensorNAS.Blocks.SubBlocks.ResNet.ResNetMiddleBlock import (
            Args as ResNetMidBlockArgs,
        )

        blocks = []
        blocks.append(
            MidBlock(
                input_shape=input_shape,
                parent_block=self,
                args={ResNetMidBlockArgs.FILTERS: 16},
            )
        )

        for i in range(1, 3):
            blocks.append(
                MidBlock(
                    input_shape=blocks[-1].get_output_shape(),
                    parent_block=self,
                    args={ResNetMidBlockArgs.FILTERS: 2 ** i * 16},
                )
            )

        return blocks

    def generate_sub_block(self, input_shape, subblock_type):

        if subblock_type == self.SubBlocks.RESNET_MIDDLE_BLOCK:
            from TensorNAS.Blocks.SubBlocks.ResNet.ResNetMiddleBlock import (
                Block as MidBlock,
            )

            return [MidBlock(input_shape=input_shape, parent_block=self)]
