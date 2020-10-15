from enum import Enum, auto

from tensornas.blocktemplates.subblocks.classificationblock import ClassificationBlock
from tensornas.blocktemplates.subblocks.featureextractionblock import (
    FeatureExtractionBlock,
)
from tensornas.core.blockarchitecture import BlockArchitecture


class ClassificationArchitectureSubBlocks(Enum):
    FEATURE_EXTRACTION_BLOCK = auto()


class ClassificationBlockArchitecture(BlockArchitecture):
    MAX_SUB_BLOCKS = 5
    SUB_BLOCK_TYPES = ClassificationArchitectureSubBlocks

    def __init__(self, input_shape, class_count):
        self.class_count = class_count

        super().__init__(input_shape, parent_block=None)

    def validate(self):
        ret = True
        if not isinstance(self.sub_blocks[-1], ClassificationBlock):
            ret = False
        return ret

    def generate_constrained_input_sub_blocks(self, input_shape):
        pass

    def generate_constrained_output_sub_blocks(self, input_shape):
        self.sub_blocks.append(
            ClassificationBlock(
                input_shape=self.input_shape,
                parent_block=self,
                class_count=self.class_count,
            )
        )

    def mutate(self):
        pass

    def generate_random_sub_block(self, input_shape, layer_type):
        if layer_type == self.SUB_BLOCK_TYPES.FEATURE_EXTRACTION_BLOCK.value:
            return FeatureExtractionBlock(input_shape=input_shape, parent_block=self)
