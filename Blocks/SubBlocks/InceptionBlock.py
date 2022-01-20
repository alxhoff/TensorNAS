from TensorNAS.Core.Block import Block

"""
An inception block is designed to make the model wider instead of deeper. Thus an inception block is responsible
for taking it's SubBlocks and making them parallel to each other.
"""


class Block(Block):
    """
    Layers that can be used in the extraction of features
    """

    MAX_SUB_BLOCKS = 4
    MIN_SUB_BLOCKS = 2

    from enum import Enum

    class SubBlocks(Enum):
        from enum import auto

        FILTER_BANK = auto()

    def generate_random_sub_block(self, input_shape, layer_type):
        from TensorNAS.Blocks.SubBlocks.FilterBankBlock import (
            Block as FilterBankBlock,
        )

        if layer_type == self.SubBlocks.FILTER_BANK:
            return [FilterBankBlock(input_shape=input_shape, parent_block=self)]

    def get_keras_layers(self, input_tensor):
        import tensorflow as tf

        filter_banks = [sb.get_keras_layers(input_tensor) for sb in self.middle_blocks]
        if len(filter_banks) > 1:
            return tf.keras.layers.Concatenate()(filter_banks)
        else:
            return filter_banks[0]
