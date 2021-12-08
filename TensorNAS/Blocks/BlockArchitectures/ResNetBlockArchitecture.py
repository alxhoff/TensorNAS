from TensorNAS.Core.BlockArchitecture import ClassificationBlockArchitecture


class Block(ClassificationBlockArchitecture):

    MAX_SUB_BLOCKS = 1
    MIN_SUB_BLOCKS = 1

    from enum import Enum

    class SubBlocks(Enum):
        from enum import auto

        NONE = auto()

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

    def generate_random_sub_block(self, input_shape, subblock_type):
        from TensorNAS.Blocks.SubBlocks.ResNet.ResNetMiddleBlockArray import (
            Block as MidBlock,
        )

        return [MidBlock(input_shape=input_shape, parent_block=self)]
