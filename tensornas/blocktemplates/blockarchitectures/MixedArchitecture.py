from enum import Enum, auto

from tensornas.blocktemplates.subblocks.TwoDClassificationBlock import (
    TwoDClassificationBlock,
)
from tensornas.blocktemplates.subblocks.FeatureExtractionBlock import (
    FeatureExtractionBlock,
)
from tensornas.blocktemplates.subblocks.EffNetBlock import (
    EffNetBlock,
)

from tensornas.blocktemplates.subblocks.FireBlock import (
    FireBlock,
)
from tensornas.blocktemplates.subblocks.GhostBlock import (
    GhostBlock,
)
from tensornas.blocktemplates.subblocks.InceptionBlock import (
    InceptionBlock,
)
from tensornas.blocktemplates.subblocks.ExpandBlock import (
    ExpandBlock,
)
from tensornas.blocktemplates.subblocks.MobilNetBlock import (
    MobileNetBlock,
)
from tensornas.blocktemplates.subblocks.ResidualBlock import(
    ResidualBlock,
)
from tensornas.blocktemplates.subblocks.ShuffleNetBlock import(
    ShuffleNetBlock,
)
from tensornas.blocktemplates.subblocks.SqueezeExpansionBLock import(
    SqueezeExpansionBlock,
)

from tensornas.core.blockarchitecture import BlockArchitecture


class MixedArchitectureSubBlocks(Enum):
    FEATURE_EXTRACTION_BLOCK = auto()
    CLASSIFICATION_BLOCK = auto()
    FIRE_BLOCK=auto()
    GHOST_BLOCK= auto()
    INCEPTION_BLOCK=auto()
    EXPAND_BLOCK=auto()
    MOBILE_NET_BLOCK=auto()
    RESIDUAL_BLOCK=auto()
    SHUFFLE_NET_BLOCK=auto()
    SQUEEZE_EXPNASION_BLOCK=auto()


class MixedBlockArchitecture(BlockArchitecture):
    MAX_SUB_BLOCKS = 2

    SUB_BLOCK_TYPES = MixedArchitectureSubBlocks

    def __init__(self, input_shape, class_count):
        self.class_count = class_count

        super().__init__(input_shape, parent_block=None, layer_type=None)

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
        if layer_type == self.SUB_BLOCK_TYPES.FEATURE_EXTRACTION_BLOCK:
            return [
                FeatureExtractionBlock(
                    input_shape=input_shape, parent_block=self, layer_type=layer_type
                )
            ]
        if layer_type == self.SUB_BLOCK_TYPES.MOBILE_NET_BLOCK:
            return [
                MobileNetBlock(
                    input_shape=input_shape, parent_block=self, layer_type=layer_type
                )
            ]
        if layer_type == self.SUB_BLOCK_TYPES.SHUFFLE_NET_BLOCK:
            return [
                ShuffleNetBlock(
                    input_shape=input_shape, parent_block=self, layer_type=layer_type
                )
            ]
        if layer_type == self.SUB_BLOCK_TYPES.FIRE_BLOCK:
            return [
                FireBlock(
                    input_shape=input_shape, parent_block=self, layer_type=layer_type
                )
            ]
        if layer_type == self.SUB_BLOCK_TYPES.GHOST_BLOCK:
            return [
                GhostBlock(
                    input_shape=input_shape, parent_block=self, layer_type=layer_type
                )
            ]
        if layer_type == self.SUB_BLOCK_TYPES.INCEPTION_BLOCK:
            return [
                InceptionBlock(
                    input_shape=input_shape, parent_block=self, layer_type=layer_type
                )
            ]
        if layer_type == self.SUB_BLOCK_TYPES.EXPAND_BLOCK:
            return [
                ExpandBlock(
                    input_shape=input_shape, parent_block=self, layer_type=layer_type
                )
            ]
        if layer_type == self.SUB_BLOCK_TYPES.RESIDUAL_BLOCK:
            return [
                ResidualBlock(
                    input_shape=input_shape, parent_block=self, layer_type=layer_type
                )
            ]
        return []
