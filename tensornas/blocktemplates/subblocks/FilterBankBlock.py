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
        import random

        layers = []
        if random.randint(0, 1):
            layers = [
                LayerBlock(
                    input_shape=input_shape,
                    parent_block=self,
                    layer_type=SupportedLayers.SAMEMAXPOOL2D,
                )
            ]
        if len(layers):
            layers.append(
                LayerBlock(
                    input_shape=layers[0].get_output_shape(),
                    parent_block=self,
                    layer_type=SupportedLayers.POINTWISECONV2D,
                )
            )
        else:
            layers.append(
                LayerBlock(
                    input_shape=input_shape,
                    parent_block=self,
                    layer_type=SupportedLayers.POINTWISECONV2D,
                )
            )
        return layers

    def generate_random_sub_block(self, input_shape, layer_type):
        if layer_type == self.SUB_BLOCK_TYPES.CONV2D:
            return [
                LayerBlock(
                    input_shape=input_shape,
                    parent_block=self,
                    layer_type=SupportedLayers.SAMECONV2D,
                )
            ]
        return []

    def get_keras_layers(self, input_tensor):
        tmp = input_tensor
        for sb in self.input_blocks + self.middle_blocks + self.output_blocks:
            tmp = sb.get_keras_layers(tmp)
        return tmp
