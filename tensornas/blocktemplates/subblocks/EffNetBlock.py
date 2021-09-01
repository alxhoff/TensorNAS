from enum import Enum, auto

from tensornas.core.layerargs import *
from tensornas.core.block import Block
from tensornas.core.LayerBlock import LayerBlock
from tensornas.layers import SupportedLayers
from tensornas.layers.Conv2D import Args as conv_args
from tensornas.layers.MaxPool import Args as pool_args

# TODO what happens with an EffNet block when the input channel is odd or 1?
import tensorflow as tf


class SubBlockTypes(Enum):
    NONE = auto()


class EffNetBlock(Block):

    MAX_SUB_BLOCKS = 0
    SUB_BLOCK_TYPES = SubBlockTypes

    def generate_random_sub_block(self, input_shape, layer_type):
        return []

    def generate_constrained_input_sub_blocks(self, input_shape):

        if (input_shape[-1] % 2) == 0:
            bottleneck_factor = 2
        else:
            bottleneck_factor = 1

        layers = []
        if bottleneck_factor != 1:
            layers.append(
                LayerBlock(
                    input_shape=input_shape,
                    parent_block=self,
                    layer_type=SupportedLayers.POINTWISECONV2D,
                    args={conv_args.FILTERS: int(input_shape[-1] / bottleneck_factor)},
                )
            )
        layers.append(
            LayerBlock(
                input_shape=layers[-1].get_output_shape()
                if bottleneck_factor != 1
                else input_shape,
                parent_block=self,
                layer_type=SupportedLayers.DEPTHWISECONV2D,
                args={conv_args.KERNEL_SIZE: (1, 3)},
            )
        )
        layers.append(
            LayerBlock(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                layer_type=SupportedLayers.MAXPOOL2D,
                args={
                    pool_args.POOL_SIZE: (2, 1),
                    pool_args.STRIDES: (2, 1),
                    pool_args.PADDING: ArgPadding.SAME,
                },
            )
        )
        layers.append(
            LayerBlock(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                layer_type=SupportedLayers.DEPTHWISECONV2D,
                args={conv_args.KERNEL_SIZE: (3, 1)},
            )
        )
        if bottleneck_factor != 1:
            layers.append(
                LayerBlock(
                    input_shape=layers[-1].get_output_shape(),
                    parent_block=self,
                    layer_type=SupportedLayers.POINTWISECONV2D,
                    args={
                        conv_args.KERNEL_SIZE: (bottleneck_factor, 1),
                        conv_args.FILTERS: input_shape[-1],
                    },
                )
            )
        return layers
