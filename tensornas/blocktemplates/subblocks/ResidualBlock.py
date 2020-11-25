from enum import Enum, auto
from tensorflow import keras

from demos.DemoMNISTInput import input_shape
from tensornas.core.block import Block
from tensornas.blocktemplates.subblocks.FeatureExtractionBlock import FeatureExtractionBlock


class SubBlockTypes(Enum):
    SAME_CONV2D = auto()

def _shortcut(input, residual):
    input_shape = keras.backend.int_shape(input)
    residual_shape = keras.backend.int_shape(residual)
    stride_width = int(round(input_shape[1]/residual_shape[1]))
    stride_height = int(round(input_shape[2]/residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input

    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = keras.layers.Conv2D(filters=residual_shape[3],
                                       kernel_size=(1,1),
                                       strides=(stride_width, stride_height),
                                       padding="valid",
                                       kernel_initializer="he_normal",
                                       kernel_regularizer=keras.regularizers.l2(0.0001))(input)
        os = keras.backend.int_shape(shortcut)

    ret = keras.layers.add([shortcut, residual])
    return ret

class ResidualBlock(Block):
    MAX_SUB_BLOCKS = 2
    SUB_BLOCK_TYPES = SubBlockTypes

    def generate_random_sub_block(self, input_shape, layer_type):
        if layer_type == self.SUB_BLOCK_TYPES.SAME_CONV2D:
            return [
                FeatureExtractionBlock(
                    input_shape=input_shape, parent_block=self, layer_type=layer_type
                )
            ]
        return []

    def get_keras_layers(self, input_tensor):
        tmp = input_tensor
        for sb in self.input_blocks + self.middle_blocks + self.output_blocks:
            tmp = sb.get_keras_layers(tmp)
        # return keras.layers.Add()([input_tensor, tmp])
        return _shortcut(input_tensor, tmp)