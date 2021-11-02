from enum import Enum, auto

from TensorNAS.Core.Block import Block
from TensorNAS.Core.LayerBlock import Block as LayerBlock
from TensorNAS.Layers import SupportedLayers


class SubBlockTypes(Enum):

    BATCH_NORMILIZATION = auto()
    ACTIVATION = auto()


class Block(Block):

    MAX_SUB_BLOCKS = 1
    SUB_BLOCK_TYPES = SubBlockTypes

    def generate_constrained_output_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Activation import Args as activation_args
        from TensorNAS.Core.LayerArgs import ArgActivations

        layers = []

        layers.append(
            LayerBlock(
                input_shape=input_shape,
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

        return layers
