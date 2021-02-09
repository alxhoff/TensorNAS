from enum import Enum, auto

from tensornas.blocktemplates.subblocks.TwoDClassificationBlock import (
    TwoDClassificationBlock,
)
from tensornas.blocktemplates.subblocks.EffNetBlock import EffNetBlock
from tensornas.core.blockarchitecture import BlockArchitecture


class EffNetArchitectureSubBlocks(Enum):
    EFFNET_BLOCK = auto()
    CLASSIFICATION_BLOCK = auto()


class EffNetBlockArchitecture(BlockArchitecture):
    MAX_SUB_BLOCKS = 1
    SUB_BLOCK_TYPES = EffNetArchitectureSubBlocks

    def __init__(self, input_shape, class_count):
        self.class_count = class_count

        super().__init__(input_shape, parent_block=None, layer_type=None)

    def generate_constrained_input_sub_blocks(self, input_shape):
        return [
            EffNetBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=self.SUB_BLOCK_TYPES.EFFNET_BLOCK,
            )
        ]

    def generate_constrained_output_sub_blocks(self, input_shape):
        layers = []
        # layers.append(GlobalAveragePool2D(input_shape=input_shape))
        layers.append(
            TwoDClassificationBlock(
                input_shape=input_shape,
                parent_block=self,
                class_count=self.class_count,
                layer_type=self.SUB_BLOCK_TYPES.CLASSIFICATION_BLOCK,
            )
        )
        return layers

    def generate_random_sub_block(self, input_shape, layer_type):
        return [
            EffNetBlock(
                input_shape=input_shape, parent_block=self, layer_type=layer_type
            )
        ]
