from enum import Enum, auto

from tensornas.core.block import Block
from tensornas.core.layerblock import LayerBlock
from tensornas.layers import SupportedLayers


class ClassificationBlockLayerTypes(Enum):
    """
    Layers that can be used in the generation of a feature extraction block are enumerated here for random selection
    """

    FLATTEN = auto()
    HIDDENDENSE = auto()
    DROPOUT = auto()


class ClassificationBlock(Block):
    """
    Block used for performing classification

    An optional class_count parameter specifies if there is a known number of output classes. This would be required
    if the classification block is the final block in a model, thus responsible for the NN output.

    If the classification block is not the output then is does not necessarily have a required number of outputs,
    meaning it can be a random number
    """

    DROPOUT_RATE_MAX = 0.2

    MAX_SUB_BLOCKS = 2
    SUB_BLOCK_TYPES = ClassificationBlockLayerTypes

    def __init__(self, input_shape, parent_block, layer_type, class_count):
        self.class_count = class_count

        super().__init__(input_shape, parent_block, layer_type)

    def validate(self, repair):
        ret = True
        if not self.output_blocks[-1].layer_type == SupportedLayers.OUTPUTDENSE:
            ret = False
        return ret

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
                args=self.class_count,
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
                    args=self.DROPOUT_RATE_MAX,
                )
            ]
        return []
