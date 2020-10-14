from enum import Enum, auto

from tensornas.block import Block
from tensornas.layerblock import LayerBlock, SupportedLayerTypes


class FeatureExtractionBlockLayerTypes(Enum):
    """
    Layers that can be used in the generation of a feature extraction block are enumerated here for random selection
    """

    CONV2D = auto()
    MAXPOOL2D = auto()


class FeatureExtractionBlock(Block):
    """
    Layers that can be used in the extraction of features
    """

    MAX_SUB_BLOCKS = 10
    SUB_BLOCK_TYPES = FeatureExtractionBlockLayerTypes

    def validate(self):
        return True

    def generate_constrained_input_sub_blocks(self, input_shape):
        pass

    def generate_constrained_output_sub_blocks(self, input_shape):
        pass

    def mutate(self):
        pass

    def get_output_shape(self):
        self.sub_blocks[-1].get_output_shape()

    def generate_random_sub_block(self, input_shape, layer_type):
        if layer_type == self.SUB_BLOCK_TYPES.CONV2D.value:
            return LayerBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=SupportedLayerTypes.CONV2D,
            )
        elif layer_type == self.SUB_BLOCK_TYPES.MAXPOOL2D.value:
            return LayerBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=SupportedLayerTypes.MAX_POOL_2D,
            )
