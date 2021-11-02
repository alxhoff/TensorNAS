from enum import Enum, auto

from TensorNAS.Core.Block import Block


class DSCNNConvBlockLayerTypes(Enum):

    DEPTHWISE_CONV2D = auto()
    BATCH_NORMALIZATION_AND_ACTIVATION = auto()


class Block(Block):

    MAX_SUB_BLOCKS = 1
    SUB_BLOCK_TYPES = DSCNNConvBlockLayerTypes

    def __init__(self, input_shape, parent_block, class_count, layer_type=-1):
        self.class_count = class_count

        super().__init__(input_shape, parent_block, layer_type)

    def generate_constrained_output_sub_blocks(self, input_shape):
        from TensorNAS.Core.LayerBlock import Block as LayerBlock
        from TensorNAS.Layers import SupportedLayers
        from TensorNAS.Layers.Dense import Args as dense_args
        from TensorNAS.Core.LayerArgs import ArgActivations

        layers = []

        layers.append(
            LayerBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=SupportedLayers.DROPOUT,
            )
        )
        layers.append(
            LayerBlock(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                layer_type=SupportedLayers.AVERAGEPOOLING2D,
            )
        )
        layers.append(
            LayerBlock(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                layer_type=SupportedLayers.FLATTEN,
            )
        )
        layers.append(
            LayerBlock(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                layer_type=SupportedLayers.OUTPUTDENSE,
                args={
                    dense_args.UNITS: self.class_count,
                    dense_args.ACTIVATION: ArgActivations.SOFTMAX,
                },
            )
        )

        return layers
