from enum import Enum, auto
from TensorNAS.Core.Block import Block


class Block(Block):

    MAX_SUB_BLOCKS = 1

    class SubBlocks(Enum):

        BATCH_NORMALIZATION_AND_ACTIVATION = auto()
        CONV2D = auto()

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Conv2D.PointwiseConv2D import Layer as PointwiseConv2D
        from TensorNAS.Layers.Conv2D import Args as conv2d_args
        from TensorNAS.Core.Layer import ArgRegularizers, ArgPadding, ArgActivations

        args = {
            conv2d_args.FILTERS: 64,
            conv2d_args.PADDING: ArgPadding.SAME,
            conv2d_args.REGULARIZER: (ArgRegularizers.L2, 1e-4),
            conv2d_args.ACTIVATION: ArgActivations.NONE,
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
