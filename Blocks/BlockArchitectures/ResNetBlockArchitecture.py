from TensorNAS.Core.BlockArchitecture import ClassificationBlockArchitecture
from enum import Enum, auto


class Block(ClassificationBlockArchitecture):

    MAX_SUB_BLOCKS = 1
    MIN_SUB_BLOCKS = 1

    class SubBlocks(Enum):

        RESNET_MID_BLOCK_ARRAY = auto()

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.Blocks.SubBlocks.ResNet.ResNetInputBlock import (
            Block as InputBlock,
        )

        return [InputBlock(input_shape=input_shape, parent_block=self)]

    def generate_constrained_output_sub_blocks(self, input_shape):
        from TensorNAS.Blocks.SubBlocks.ResNet.ResNetOutputBlock import (
            Block as OutputBlock,
        )

        return [OutputBlock(input_shape=input_shape, parent_block=self)]

    def generate_constrained_middle_sub_blocks(self, input_shape):
        from TensorNAS.Blocks.SubBlocks.ResNet.ResNetMiddleBlockArray import (
            Block as MidBlockArray,
        )

        return [MidBlockArray(input_shape=input_shape, parent_block=self)]

    def generate_sub_block(self, input_shape, subblock_type):
        from TensorNAS.Blocks.SubBlocks.ResNet.ResNetMiddleBlockArray import (
            Block as MidBlockArray,
        )

        if subblock_type == self.SubBlocks.RESNET_MID_BLOCK_ARRAY:
            return [MidBlockArray(input_shape=input_shape, parent_block=self)]

        return []
