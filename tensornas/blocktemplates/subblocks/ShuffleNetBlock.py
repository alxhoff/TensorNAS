from enum import Enum, auto

from tensornas.blocktemplates.subblocks.TwoDClassificationBlock import (
    TwoDClassificationBlock,
)

from tensornas.core.layerargs import ArgActivations, ArgPadding
from tensornas.core.modelutil import shortcut
from tensornas.core.block import Block
from tensornas.core.layerblock import LayerBlock
from tensornas.layers import SupportedLayers
from tensornas.layers.Conv2D import Args as conv_args


class SubBlockTypes(Enum):
    """
    Layers that can be used in the generation of a feature extraction block are enumerated here for random selection
    """

    DEPTHWISECONV2D = auto()
    POINTWISECONV2D = auto()
    GROUPEDCONV2D = auto()


class ShuffleNetBlock(Block):
    """
    Layers that can be used in the extraction of features
    """

    # TODO the input to the layer must be divisible by 4. Maybe some thought should be put
    # into making this more robust and removing the placeholder input conv2d block.

    MAX_SUB_BLOCKS = 0
    SUB_BLOCK_TYPES = SubBlockTypes

    def generate_constrained_input_sub_blocks(self, input_shape):
        return [
            LayerBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=SupportedLayers.CONV2D,
                args={conv_args.FILTERS: 16, conv_args.PADDING: ArgPadding.SAME},
            )
        ]

    def generate_random_sub_block(self, input_shape, layer_type):
        return []

    def generate_constrained_output_sub_blocks(self, input_shape):
        residual_channel_depth = input_shape[-1]
        bottleneck_filters = residual_channel_depth // 4

        layers = []
        layers.append(
            LayerBlock(
                input_shape=input_shape,
                parent_block=self,
                layer_type=SupportedLayers.GROUPEDPOINTWISECONV2D,
                args={
                    conv_args.FILTERS: bottleneck_filters,
                    conv_args.ACTIVATION: ArgActivations.RELU,
                    conv_args.PADDING: ArgPadding.SAME,
                },
            )
        )
        layers.append(
            LayerBlock(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                layer_type=SupportedLayers.SHUFFLE,
            )
        )
        layers.append(
            LayerBlock(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                layer_type=SupportedLayers.DEPTHWISECONV2D,
                args={
                    conv_args.KERNEL_SIZE: (3, 3),
                    conv_args.PADDING: ArgPadding.SAME,
                },
            )
        )
        layers.append(
            LayerBlock(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                layer_type=SupportedLayers.GROUPEDPOINTWISECONV2D,
                args={
                    conv_args.FILTERS: residual_channel_depth,
                    conv_args.ACTIVATION: ArgActivations.RELU,
                },
            )
        )
        return layers

    def get_keras_layers(self, input_tensor):
        tmp = input_tensor
        for sb in self.input_blocks + self.middle_blocks + self.output_blocks:
            tmp = sb.get_keras_layers(tmp)
        return shortcut(input_tensor, tmp)
