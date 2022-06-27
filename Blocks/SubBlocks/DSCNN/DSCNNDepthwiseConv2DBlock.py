from enum import Enum, auto
from TensorNAS.Core.Block import Block


class Block(Block):

    MAX_SUB_BLOCKS = 0

    class SubBlocks(Enum):

        DEPTHWISE_CONV2D = auto()
        BATCH_NORMALIZATION_AND_ACTIVATION = auto()

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Conv2D.DepthwiseConv2D import Layer as DepthwiseConv2D
        from TensorNAS.Layers.Conv2D import Args as conv2d_args
        from TensorNAS.Core.Layer import ArgRegularizers
        from TensorNAS.Core.Layer import ArgPadding
        from TensorNAS.Core.Layer import ArgActivations

        args = {
            conv2d_args.FILTERS: 64,
            conv2d_args.KERNEL_SIZE: (3, 3),
            conv2d_args.PADDING: ArgPadding.SAME,
            conv2d_args.REGULARIZER: (ArgRegularizers.L2, 1e-4),
            conv2d_args.ACTIVATION: ArgActivations.NONE,
        }

        return [
            DepthwiseConv2D(
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
