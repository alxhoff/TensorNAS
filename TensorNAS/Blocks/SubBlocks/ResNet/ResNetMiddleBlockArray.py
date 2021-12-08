from TensorNAS.Core.Block import Block


class Block(Block):

    from enum import Enum

    class SubBlocks(Enum):
        from enum import auto

        RESNET_MIDDLE_BLOCK = auto()

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.Blocks.SubBlocks.ResNet.ResNetMiddleBlock import (
            Block as MidBlock,
        )

        blocks = [MidBlock(input_shape=input_shape, parent_block=self)]
        for i in range(2):
            blocks.append(
                MidBlock(input_shape=blocks[-1].get_output_shape(), parent_block=self)
            )

        return blocks

    def generate_random_sub_block(self, input_shape, subblock_type):

        if subblock_type == self.SubBlocks.RESNET_MIDDLE_BLOCK:
            from TensorNAS.Blocks.SubBlocks.ResNet.ResNetMiddleBlock import (
                Block as MidBlock,
            )

            return [MidBlock(input_shape=input_shape, parent_block=self)]
