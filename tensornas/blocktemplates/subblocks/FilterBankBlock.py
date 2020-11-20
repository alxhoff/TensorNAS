from enum import Enum, auto

from tensornas.core.block import Block
from tensornas.core.layerblock import LayerBlock
from tensornas.layers import SupportedLayers


class SubBlockTypes(Enum):
    CONV2D = auto()


class FilterBankBlock(Block):

    MAX_SUB_BLOCKS = 3
    SUB_BLOCK_TYPES = SubBlockTypes

    def generate_constrained_input_sub_blocks(self, input_shape):
        return [
            LayerBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=SupportedLayers.POINTWISECONV,
            )
        ]

    def generate_random_sub_block(self, input_shape, layer_type):
        if layer_type == self.SUB_BLOCK_TYPES.CONV2D:
            return [
                LayerBlock(
                    input_shape=input_shape,
                    parent_block=self,
                    layer_type=SupportedLayers.CONV2D,
                )
            ]
        return []

    def get_keras_layers(self):
        array = None
        for sb in self.input_blocks + self.middle_blocks + self.output_blocks:
            if not array:
                array = sb.get_keras_layers()
            else:
                array = sb.get_keras_layers()(array)
        return array
