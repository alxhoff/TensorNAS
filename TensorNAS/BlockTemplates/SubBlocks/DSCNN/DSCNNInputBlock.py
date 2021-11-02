from enum import Enum, auto

from TensorNAS.Core.Block import Block


class DSCNNConvBlockLayerTypes(Enum):

    DEPTHWISE_CONV2D = auto()
    BATCH_NORMALIZATION_AND_ACTIVATION = auto()


class Block(Block):

    MAX_SUB_BLOCKS = 1
    SUB_BLOCK_TYPES = DSCNNConvBlockLayerTypes

    def generate_random_sub_block(self, input_shape, block_type):
        from TensorNAS.Core.LayerBlock import Block as LayerBlock
        from TensorNAS.Layers import SupportedLayers
        from TensorNAS.Layers.Conv2D import Args as conv2d_args
        from TensorNAS.Layers.Activation import Args as activation_args
        from TensorNAS.Layers.Dropout import Args as dropout_args
        from TensorNAS.Core.LayerArgs import ArgPadding, ArgRegularizers, ArgActivations

        layers = []

        layers.append(
            LayerBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=SupportedLayers.CONV2D,
                args={
                    conv2d_args.FILTERS: 64,
                    conv2d_args.KERNEL_SIZE: (10, 4),
                    conv2d_args.STRIDES: (2, 2),
                    conv2d_args.KERNEL_REGULARIZER: (ArgRegularizers.L2, 1e-4),
                    conv2d_args.PADDING: ArgPadding.SAME,
                },
            )
        )
        layers.append(
            LayerBlock(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                layer_type=SupportedLayers.BATCHNORMALIZATION,
            )
        )
        layers.append(
            LayerBlock(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                layer_type=SupportedLayers.ACTIVATION,
                args={activation_args.ACTIVATION: ArgActivations.RELU},
            )
        )
        layers.append(
            LayerBlock(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                layer_type=SupportedLayers.DROPOUT,
                args={dropout_args.RATE: 0.2},
            )
        )

        return layers
