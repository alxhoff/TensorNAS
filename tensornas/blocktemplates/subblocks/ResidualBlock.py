from tensornas.core.layerblock import LayerBlock
from tensornas.core.block import Block
from enum import Enum, auto
from tensornas.layers import SupportedLayers
from tensorflow.keras.layers import Add


class SubBlockTypes(Enum):
    CONV2D = auto()


class ResidualBlock(Block):
    MAX_SUB_BLOCKS = 2
    SUB_BLOCK_TYPES = SubBlockTypes

    def generate_random_sub_block(self, input_shape, layer_type):
        X_shortcut = input_shape
        for i in range(self.MAX_SUB_BLOCKS):
            if layer_type == self.SUB_BLOCK_TYPES.CONV2D:
                X = LayerBlock(
                    input_shape=input_shape,
                    parent_block=self,
                    layer_type=SupportedLayers.CONV2D,
                ).get_keras_layers()

        X = LayerBlock(
            input_shape=input, parent_block=self, layer_type=SupportedLayers.CONV2D,
        ).get_keras_layers()
        X = Add()([X, X_shortcut])
        return [X]
