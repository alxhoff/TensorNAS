from TensorNAS.Core.BlockArchitecture import ClassificationBlockArchitecture
from enum import Enum, auto


class Block(ClassificationBlockArchitecture):

    MAX_SUB_BLOCKS = 6
    MIN_SUB_BLOCKS = 1

    class SubBlocks(Enum):

        EFFNET_BLOCK = auto()
        EXPAND_BLOCK = auto()
        FEATURE_EXTRACTION_BLOCK = auto()
        FIRE_BLOCK = auto()
        GHOST_BLOCK = auto()
        MOBILE_NET_BLOCK = auto()
        RESIDUAL_BLOCK = auto()
        SHUFFLE_NET_BLOCK = auto()
        SQUEEZE_EXPANSION_BLOCK = auto()
        TWOD_CLASSIFICATION_BLOCK = auto()

    def generate_constrained_output_sub_blocks(self, input_shape):
        from TensorNAS.Blocks.SubBlocks.TwoDClassificationBlock import (
            Block as TwoDClassificationBlock,
        )

        return [
            TwoDClassificationBlock(
                input_shape=input_shape,
                parent_block=self,
                class_count=self.class_count,
            )
        ]

    def generate_sub_block(self, input_shape, subblock_type, args=None):

        from TensorNAS.Blocks.SubBlocks.EffNetBlock import Block as EffNetBlock
        from TensorNAS.Blocks.SubBlocks.ExpandBlock import Block as ExpandBlock
        from TensorNAS.Blocks.SubBlocks.FeatureExtractionBlock import (
            Block as FeatureExtractionBlock,
        )
        from TensorNAS.Blocks.SubBlocks.FireBlock import Block as FireBlock
        from TensorNAS.Blocks.SubBlocks.GhostBlock import Block as GhostBlock
        from TensorNAS.Blocks.SubBlocks.MobilNetMidBlock import Block as MobileNetBlock
        from TensorNAS.Blocks.SubBlocks.ResidualBlock import Block as ResidualBlock
        from TensorNAS.Blocks.SubBlocks.ShuffleNetBlock import Block as ShuffleNetBlock

        if subblock_type == self.SubBlocks.EFFNET_BLOCK:
            return [EffNetBlock(input_shape=input_shape, parent_block=self)]
        elif subblock_type == self.SubBlocks.EXPAND_BLOCK:
            return [ExpandBlock(input_shape=input_shape, parent_block=self)]
        elif subblock_type == self.SubBlocks.FEATURE_EXTRACTION_BLOCK:
            return [FeatureExtractionBlock(input_shape=input_shape, parent_block=self)]
        elif subblock_type == self.SubBlocks.FIRE_BLOCK:
            return [FireBlock(input_shape=input_shape, parent_block=self)]
        elif subblock_type == self.SubBlocks.GHOST_BLOCK:
            return [GhostBlock(input_shape=input_shape, parent_block=self)]
        elif subblock_type == self.SubBlocks.MOBILE_NET_BLOCK:
            return [MobileNetBlock(input_shape=input_shape, parent_block=self)]
        elif subblock_type == self.SubBlocks.RESIDUAL_BLOCK:
            return [ResidualBlock(input_shape=input_shape, parent_block=self)]
        elif subblock_type == self.SubBlocks.SHUFFLE_NET_BLOCK:
            return [ShuffleNetBlock(input_shape=input_shape, parent_block=self)]

        return []
