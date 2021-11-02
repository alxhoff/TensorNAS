from enum import Enum, auto

from TensorNAS.Core.Block import Block


class DSCNNConvBlockLayerTypes(Enum):

    BATCH_NORMALIZATION_AND_ACTIVATION = auto()
    CONV2D = auto()


class Block(Block):

    MAX_SUB_BLOCKS = 1
    SUB_BLOCK_TYPES = DSCNNConvBlockLayerTypes

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.Core.LayerBlock import Block as LayerBlock
        from TensorNAS.Layers import SupportedLayers
        from TensorNAS.Layers.Conv2D import Args as conv2d_args
        from TensorNAS.Core.LayerArgs import ArgPadding, ArgRegularizers

        args = {
            conv2d_args.FILTERS: 64,
            conv2d_args.PADDING: ArgPadding.SAME,
            conv2d_args.KERNEL_REGULARIZER: (ArgRegularizers.L2, 1e-4),
        }

        return [
            LayerBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=SupportedLayers.CONV2D,
                args=args,
            )
        ]

    def generate_constrained_output_sub_blocks(self, input_shape):

        from TensorNAS.BlockTemplates.SubBlocks.DSCNN.DSCNNNormActivationBlock import (
            Block as NABlock,
        )

        return [
            NABlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=self.SUB_BLOCK_TYPES.BATCH_NORMALIZATION_AND_ACTIVATION,
            )
        ]
