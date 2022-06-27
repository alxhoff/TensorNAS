from TensorNAS.Core.Block import Block
from enum import Enum, auto


class Block(Block):

    MAX_SUB_BLOCKS = 0
    MIN_SUB_BLOCKS = 0

    class SubBlocks(Enum):

        AVERAGE_POOL2D = auto()
        OUTPUT_CLASSIFICATION = auto()

    def generate_constrained_output_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Pool.AveragePooling2D import Layer as AveragePooling2D
        from TensorNAS.Blocks.SubBlocks.TwoDClassificationBlock import (
            Block as TwoDClassificationBlock,
        )

        from TensorNAS.Layers.Pool import Args as pool_args

        args = {pool_args.STRIDES: None, pool_args.POOL_SIZE: input_shape[0:2]}

        layers = []
        layers.append(
            AveragePooling2D(input_shape=input_shape, parent_block=self, args=args)
        )
        layers.append(
            TwoDClassificationBlock(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                class_count=self.parent_block.class_count,
            )
        )

        return layers

    def generate_sub_block(self, input_shape, layer_type, args=None):

        if layer_type == self.SubBlocks.AVERAGE_POOL2D:
            from TensorNAS.Layers.Pool.AveragePooling2D import Layer as AveragePooling2D

            return [AveragePooling2D(input_shape=input_shape, parent_block=self)]
        elif layer_type == self.SubBlocks.OUTPUT_CLASSIFICATION:
            from TensorNAS.Blocks.SubBlocks.TwoDClassificationBlock import (
                Block as TwoDClassificationBlock,
            )

            return [
                TwoDClassificationBlock(
                    input_shape=input_shape,
                    parent_block=self,
                    class_count=self.parent_block.class_count,
                )
            ]

        return []
