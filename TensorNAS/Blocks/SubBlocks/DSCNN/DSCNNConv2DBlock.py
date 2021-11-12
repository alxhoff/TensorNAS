from enum import Enum, auto

from TensorNAS.Core.Block import Block


class Block(Block):

    MAX_SUB_BLOCKS = 1

    from enum import Enum

    class SubBlocks(Enum):
        from enum import auto

        BATCH_NORMALIZATION_AND_ACTIVATION = auto()
        CONV2D = auto()

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Conv2D.PointwiseConv2D import Layer as PointwiseConv2D
        from TensorNAS.Layers.Conv2D import Args as conv2d_args
        from TensorNAS.Core.Layer import ArgRegularizers
        from TensorNAS.Core.Layer import ArgPadding

        args = {
            conv2d_args.FILTERS: 64,
            conv2d_args.PADDING: ArgPadding.SAME,
            conv2d_args.KERNEL_REGULARIZER: (ArgRegularizers.L2, 1e-4),
        }

        return [
            PointwiseConv2D(
                input_shape=input_shape,
                parent_block=self,
                args=args,
            )
        ]

    def generate_constrained_output_sub_blocks(self, input_shape):

        from TensorNAS.Blocks.SubBlocks.DSCNN.DSCNNNormActivationBlock import (
            Block as NABlock,
        )

        return [
            NABlock(
                input_shape=input_shape,
                parent_block=self,
            )
        ]
