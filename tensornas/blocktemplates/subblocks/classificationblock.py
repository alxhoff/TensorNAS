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

    MAX_SUB_BLOCKS = 10
    SUB_BLOCK_TYPES = ClassificationBlockLayerTypes

    def __init__(self, input_shape, parent_block, class_count):
        self.class_count = class_count

        super().__init__(input_shape, parent_block)

    def validate(self):
        ret = True
        if (
            not self.sub_blocks[-1].layer.get_name()
            == SupportedLayers.OUTPUTDENSE.value
        ):
            ret = False
        return ret

    def generate_constrained_input_sub_blocks(self, input_shape):
        pass

    def generate_constrained_output_sub_blocks(self, input_shape):
        self.sub_blocks.append(
            LayerBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=SupportedLayers.OUTPUTDENSE,
                args=self.class_count,
            )
        )

    def mutate(self):
        pass

    def generate_random_sub_block(self, input_shape, layer_type):
        if layer_type == self.SUB_BLOCK_TYPES.FLATTEN.value:
            return LayerBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=SupportedLayers.FLATTEN,
            )
        elif layer_type == self.SUB_BLOCK_TYPES.HIDDENDENSE.value:
            return LayerBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=SupportedLayers.HIDDENDENSE,
            )
        elif layer_type == self.SUB_BLOCK_TYPES.DROPOUT.value:
            return LayerBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=SupportedLayers.DROPOUT,
                args=self.DROPOUT_RATE_MAX,
            )
