from tensornas.core.layerblock import LayerBlock
from tensornas.core.block import Block
from enum import Enum, auto
from tensornas.layers import SupportedLayers


class SubBlockTypes(Enum):
    SEPARABLE_CONV = auto()


class MobileNetBlock(Block):
    MAX_SUB_BLOCKS = 2
    SUB_BLOCK_TYPES = SubBlockTypes

    # input constrained layer for SqueezeBlock: 1x1xN_c Conv layer

    def generate_random_sub_block(self, input_shape, layer_type):
        if layer_type == self.SUB_BLOCK_TYPES.SEPARABLE_CONV:
            return [
                LayerBlock(
                    input_shape=input_shape,
                    parent_block=self,
                    layer_type=SupportedLayers.SEPARABLECONV2D,
                )
            ]
