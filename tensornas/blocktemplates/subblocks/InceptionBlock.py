# class to create Expansion block layers as objects and add constraints on input
# fixed layers and output fixed layers for the Expansion Block

from enum import Enum, auto
import tensorflow as tf

from tensornas.core.block import Block
from tensornas.core.layerblock import LayerBlock
from tensornas.layers import SupportedLayers

"""
An inception block is designed to make the model wider instead of deeper. Thus an inception block is responsible
for taking it's subblocks and making them parallel to each other.
"""

class InceptionBlockLayerTypes(Enum):
    """
    Layers that can be used in the generation of a feature extraction block are enumerated here for random selection
    """

    CONV2D = auto()
    POINTWISECONV= auto()
    MAXPOOL2D = auto()


class InceptionBlock(Block):
    """
    Layers that can be used in the extraction of features
    """

    MAX_SUB_BLOCKS = 3
    SUB_BLOCK_TYPES = InceptionBlockLayerTypes


    def generate_random_sub_block(self, input_shape, layer_type):
        #if layer_type == self.SUB_BLOCK_TYPES.POINTWISECONV:
        Conv_0= LayerBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=SupportedLayers.CONV2D,
            )
        Conv_0_branch_0= Conv_0.get_keras_layers()
        #elif layer_type == self.SUB_BLOCK_TYPES.CONV2D:
        Conv_1= LayerBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=SupportedLayers.CONV2D,
            )
        Conv_1_branch_1=Conv_1.get_keras_layers()
        Conv_2=LayerBlock(
            input_shape=Conv_1.get_output_shape(),
            parent_block=self,
            layer_type=SupportedLayers.CONV2D,
        )
        Conv_2_branch_1=Conv_2.get_keras_layers()
        Conv_3=LayerBlock(
            input_shape=input_shape,
            parent_block=self,
            layer_type=SupportedLayers.CONV2D,
        )
        Conv_3_branch_2=Conv_3.get_keras_layers()
        Conv_4 = LayerBlock(
            input_shape=Conv_3.get_output_shape(),
            parent_block=self,
            layer_type=SupportedLayers.CONV2D,
        )
        Conv_4_branch_2 = Conv_4.get_keras_layers()
        Pool_1= LayerBlock(
            input_shape=input_shape,
            parent_block=self,
            layer_type=SupportedLayers.MAXPOOL2D,
        )
        Pool_1_branch_3=Pool_1.get_keras_layers()
        Conv_5=LayerBlock(
            input_shape=Pool_1.get_output_shape(),
            parent_block=self,
            layer_type=SupportedLayers.CONV2D,
        )
        Conv_5_branch_3 = Conv_5.get_keras_layers()

        Block= tf.keras.layers.Concatenate([Conv_0_branch_0, Conv_1_branch_1, Conv_4_branch_2, Conv_5_branch_3])

        #should return a layer block object whose output shape is calculated in Layer_Block-->Block class
        return [Block]

