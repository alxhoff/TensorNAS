from enum import Enum, auto

from TensorNAS.Core.Block import Block
from TensorNAS.Core.LayerBlock import Block as LayerBlock
from TensorNAS.Layers import SupportedLayers

from TensorNAS.Layers.Dense import Args as dense_args


class SqueezeExpansionBlockLayerTypes(Enum):

    """Contains Global Average Pool layer and then Fully connected Layers with ReLu types and output dense layer with sigmoid activation function"""

    GLOBAL_AVERAGE_POOLING2D = auto()
    HIDDENDENSE = auto()
    OUTPUTDENSE = auto()


class Block(Block):

    MAX_SUB_BLOCKS = 2
    SUB_BLOCK_TYPES = SqueezeExpansionBlockLayerTypes

    def __init__(self, input_shape, parent_block, class_count, layer_type=-1):
        self.class_count = class_count

        super().__init__(input_shape, parent_block, layer_type)

    def generate_constrained_input_sub_blocks(self, input_shape):
        # TODO do not make it manually append but instead return a list of blocks
        return [
            LayerBlock(
                input_shape=None,
                parent_block=self,
                layer_type=SupportedLayers.GLOBALAVERAGEPOOL2D,
            )
        ]

    def generate_constrained_output_sub_blocks(self, input_shape):
        """Use of input_shape=None causes the input shape to be resolved from the previous layer."""
        return [
            LayerBlock(
                input_shape=None,
                parent_block=self,
                layer_type=SupportedLayers.OUTPUTDENSE,
                args={dense_args.UNITS: self.class_count},
            )
        ]

    def generate_random_sub_block(self, input_shape, layer_type):
        return [
            LayerBlock(
                input_shape=None,
                parent_block=self,
                layer_type=SupportedLayers.HIDDENDENSE,
                args={dense_args.UNITS: self.class_count},
            )
        ]
