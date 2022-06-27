from TensorNAS.Core.Block import Block
from enum import Enum, auto


class Block(Block):

    MAX_SUB_BLOCKS = 1

    class SubBlocks(Enum):

        SAME_CONV2D = auto()

    def generate_sub_block(self, input_shape, layer_type):
        from TensorNAS.Blocks.SubBlocks.FeatureExtractionBlock import (
            Block as FeatureExtractionBlock,
        )

        if layer_type == self.SubBlocks.SAME_CONV2D:
            return [FeatureExtractionBlock(input_shape=input_shape, parent_block=self)]
        return []

    def get_keras_layers(self, input_tensor):
        from TensorNAS.Tools.TensorFlow import shortcut

        tmp = input_tensor
        for sb in self.input_blocks + self.middle_blocks + self.output_blocks:
            tmp = sb.get_keras_layers(tmp)
        return shortcut(input_tensor, tmp)
