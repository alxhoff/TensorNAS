from TensorNAS.Core.BlockArchitecture import ClassificationBlockArchitecture
from enum import Enum, auto


class Block(ClassificationBlockArchitecture):

    MAX_SUB_BLOCKS = 3

    class SubBlocks(Enum):

        MOBILE_INPUT_BLOCK = auto()
        MOBILE_MID_BLOCK = auto()
        MOBILE_OUTPUT_BLOCK = auto()

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.Blocks.SubBlocks.MobileNet.MobileNetInputBlock import (
            Block as MobileNetInputBlock,
        )

        return [
            MobileNetInputBlock(
                input_shape=input_shape,
                parent_block=self,
            )
        ]

    def generate_constrained_output_sub_blocks(self, input_shape):
        from TensorNAS.Blocks.SubBlocks.MobileNet.MobileNetOutputBlock import (
            Block as MobileNetOutputBlock,
        )

        return [
            MobileNetOutputBlock(
                input_shape=input_shape,
                parent_block=self,
            )
        ]

    def generate_constrained_middle_sub_blocks(self, input_shape, args=None):
        from TensorNAS.Blocks.SubBlocks.MobileNet.MobilNetMidBlock import (
            Block as MobileNetMidBlock,
        )
        from TensorNAS.Layers.Conv2D.SeparableConv2D import Args as sep_conv_args

        layers = []

        # 1 x 16x filter
        layers.append(
            MobileNetMidBlock(
                input_shape=input_shape,
                parent_block=self,
                args={sep_conv_args.FILTERS: 16, sep_conv_args.STRIDES: (1, 1)},
            )
        )

        # 2x 32x filter
        layers.append(
            MobileNetMidBlock(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                args={sep_conv_args.FILTERS: 32, sep_conv_args.STRIDES: (2, 2)},
            )
        )

        layers.append(
            MobileNetMidBlock(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                args={sep_conv_args.FILTERS: 32, sep_conv_args.STRIDES: (1, 1)},
            )
        )

        # 2x 64x filter
        layers.append(
            MobileNetMidBlock(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                args={sep_conv_args.FILTERS: 64, sep_conv_args.STRIDES: (2, 2)},
            )
        )

        layers.append(
            MobileNetMidBlock(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                args={sep_conv_args.FILTERS: 64, sep_conv_args.STRIDES: (1, 1)},
            )
        )

        # 6x 128x filter
        layers.append(
            MobileNetMidBlock(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                args={sep_conv_args.FILTERS: 128, sep_conv_args.STRIDES: (2, 2)},
            )
        )

        for i in range(5):
            layers.append(
                MobileNetMidBlock(
                    input_shape=layers[-1].get_output_shape(),
                    parent_block=self,
                    args={sep_conv_args.FILTERS: 128, sep_conv_args.STRIDES: (1, 1)},
                )
            )

        # 2x 256x filter
        layers.append(
            MobileNetMidBlock(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                args={sep_conv_args.FILTERS: 256, sep_conv_args.STRIDES: (2, 2)},
            )
        )

        layers.append(
            MobileNetMidBlock(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                args={sep_conv_args.FILTERS: 256, sep_conv_args.STRIDES: (1, 1)},
            )
        )

        return layers

    def generate_sub_block(self, input_shape, layer_type):

        if layer_type == self.SubBlocks.MOBILE_INPUT_BLOCK:
            return [self.generate_constrained_input_sub_blocks(input_shape=input_shape)]
        elif layer_type == self.SubBlocks.MOBILE_MID_BLOCK:
            return [
                self.generate_constrained_middle_sub_blocks(input_shape=input_shape)
            ]
        elif layer_type == self.SubBlocks.MOBILE_OUTPUT_BLOCK:
            return [
                self.generate_constrained_output_sub_blocks(input_shape=input_shape)
            ]

        return []
