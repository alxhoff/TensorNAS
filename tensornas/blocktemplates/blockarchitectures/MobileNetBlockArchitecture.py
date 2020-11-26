from enum import Enum, auto

from tensornas.blocktemplates.subblocks.TwoDClassificationBlock import (
    TwoDClassificationBlock,
)
from tensornas.blocktemplates.subblocks.MobilNetBlock import MobileNetBlock
from tensornas.core.blockarchitecture import BlockArchitecture


class MobileNetArchitectureSubBlocks(Enum):
    MOBILE_BLOCK = auto()
    CLASSIFICATION_BLOCK = auto()


class MobileNetBlockArchitecture(BlockArchitecture):
    MAX_SUB_BLOCKS = 3
    SUB_BLOCK_TYPES = MobileNetArchitectureSubBlocks

    def __init__(self, input_shape, class_count):
        self.class_count = class_count

        super().__init__(input_shape, parent_block=None, layer_type=None)

    def validate(self, repair):
        ret = True
        """should add a squeezeBlock for auto selection to tensornas.blocktemplates.subblocks """
        if not isinstance(self.input_blocks[0], MobileNetBlock):
            ret = False

        return ret

    def generate_constrained_input_sub_blocks(self, input_shape):
        return [
            MobileNetBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=self.SUB_BLOCK_TYPES.MOBILE_BLOCK,
            )
        ]

    def generate_constrained_output_sub_blocks(self, input_shape):
        return [
            TwoDClassificationBlock(
                input_shape=input_shape,
                parent_block=self,
                class_count=self.class_count,
                layer_type=self.SUB_BLOCK_TYPES.CLASSIFICATION_BLOCK,
            )
        ]

    def generate_random_sub_block(self, input_shape, layer_type):
        return [
            MobileNetBlock(
                input_shape=input_shape, parent_block=self, layer_type=layer_type
            )
        ]
