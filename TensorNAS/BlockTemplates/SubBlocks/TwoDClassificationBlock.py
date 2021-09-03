from enum import Enum, auto

from TensorNAS.Core.Block import Block
from TensorNAS.Core.LayerBlock import Block as LayerBlock
from TensorNAS.Layers import SupportedLayers

from TensorNAS.Layers.Dropout import Args as dropout_args
from TensorNAS.Layers.Dense import Args as dense_args


class TwoDClassificationBlockLayerTypes(Enum):
    """
    Layers that can be used in the generation of a feature extraction block are enumerated here for random selection
    """

    FLATTEN = auto()
    HIDDENDENSE = auto()
    DROPOUT = auto()


class Block(Block):
    """
    Block used for performing classification

    An optional class_count parameter specifies if there is a known number of output classes. This would be required
    if the classification block is the final block in a model, thus responsible for the NN output.

    If the classification block is not the output then is does not necessarily have a required number of outputs,
    meaning it can be a random number
    """

    DROPOUT_RATE_MAX = 0.2

    MAX_SUB_BLOCKS = 2
    SUB_BLOCK_TYPES = TwoDClassificationBlockLayerTypes

    def __init__(self, input_shape, parent_block, class_count, layer_type=-1):
        self.class_count = class_count

        super().__init__(input_shape, parent_block, layer_type)

    def generate_constrained_input_sub_blocks(self, input_shape):
        # TODO do not make it manually append but instead return a list of blocks
        return [
            LayerBlock(
                input_shape=None, parent_block=self, layer_type=SupportedLayers.FLATTEN
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

    def check_next_layer_type(self, prev_layer_type, next_layer_type):
        if (
            prev_layer_type == SupportedLayers.FLATTEN
            and next_layer_type == self.SUB_BLOCK_TYPES.FLATTEN
        ):
            return False
        elif (
            prev_layer_type == SupportedLayers.DROPOUT
            and next_layer_type == self.SUB_BLOCK_TYPES.DROPOUT
        ):
            return False
        return True

    def generate_random_sub_block(self, input_shape, layer_type):
        if layer_type == self.SUB_BLOCK_TYPES.FLATTEN:
            return [
                LayerBlock(
                    input_shape=input_shape,
                    parent_block=self,
                    layer_type=SupportedLayers.FLATTEN,
                )
            ]
        elif layer_type == self.SUB_BLOCK_TYPES.HIDDENDENSE:
            return [
                LayerBlock(
                    input_shape=input_shape,
                    parent_block=self,
                    layer_type=SupportedLayers.HIDDENDENSE,
                )
            ]
        elif layer_type == self.SUB_BLOCK_TYPES.DROPOUT:
            return [
                LayerBlock(
                    input_shape=input_shape,
                    parent_block=self,
                    layer_type=SupportedLayers.DROPOUT,
                    args={dropout_args.RATE: self.DROPOUT_RATE_MAX},
                )
            ]
        return []
