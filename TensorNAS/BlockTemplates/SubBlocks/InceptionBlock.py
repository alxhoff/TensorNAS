# class to create Expansion block Layers as objects and add constraints on input
# fixed Layers and output fixed Layers for the Expansion Block

from enum import Enum, auto
import tensorflow as tf

from TensorNAS.Core.Block import Block
from TensorNAS.BlockTemplates.SubBlocks.FilterBankBlock import Block as FilterBankBlock

"""
An inception block is designed to make the model wider instead of deeper. Thus an inception block is responsible
for taking it's SubBlocks and making them parallel to each other.
"""


class SubBlockTypes(Enum):
    """
    Layers that can be used in the generation of a feature extraction block are enumerated here for random selection
    """

    FILTER_BANK = auto()


class Block(Block):
    """
    Layers that can be used in the extraction of features
    """

    MAX_SUB_BLOCKS = 4
    MIN_SUB_BLOCKS = 2
    SUB_BLOCK_TYPES = SubBlockTypes

    def generate_random_sub_block(self, input_shape, layer_type):
        if layer_type == self.SUB_BLOCK_TYPES.FILTER_BANK:
            return [
                FilterBankBlock(
                    input_shape=input_shape, parent_block=self, layer_type=layer_type
                )
            ]

    def get_keras_layers(self, input_tensor):
        filter_banks = [sb.get_keras_layers(input_tensor) for sb in self.middle_blocks]
        if len(filter_banks) > 1:
            return tf.keras.layers.Concatenate()(filter_banks)
        else:
            return filter_banks[0]
