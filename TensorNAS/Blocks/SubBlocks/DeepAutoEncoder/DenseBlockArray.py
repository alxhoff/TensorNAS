from TensorNAS.Core.Block import Block


class Block(Block):

    from enum import Enum

    class SubBlocks(Enum):
        from enum import auto

        DENSE_BLOCK = auto()

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.Blocks.SubBlocks.DeepAutoEncoder.DenseBlock import (
            Block as DenseBlock,
        )

        layers = [DenseBlock(input_shape=input_shape, parent_block=self, units=128)]

        for i in range(3):
            layers.append(
                DenseBlock(
                    input_shape=layers[-1].get_output_shape(),
                    parent_block=self,
                    units=128,
                )
            )

        layers.append(
            DenseBlock(
                input_shape=layers[-1].get_output_shape(), parent_block=self, units=8
            )
        )

        for i in range(4):
            layers.append(
                DenseBlock(
                    input_shape=layers[-1].get_output_shape(),
                    parent_block=self,
                    units=128,
                )
            )

        return layers
