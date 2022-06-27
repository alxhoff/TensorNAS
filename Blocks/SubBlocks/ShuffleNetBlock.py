from TensorNAS.Core.Block import Block
from enum import Enum, auto


class Block(Block):
    """
    Layers that can be used in the extraction of features
    """

    # TODO the input to the layer must be divisible by 4. Maybe some thought should be put
    # into making this more robust and removing the placeholder input conv2d block.

    MAX_SUB_BLOCKS = 0

    class SubBlocks(Enum):

        DEPTHWISECONV2D = auto()
        POINTWISECONV2D = auto()
        GROUPEDCONV2D = auto()

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Conv2D.Conv2D import Layer as Conv2D
        from TensorNAS.Core.Layer import ArgPadding
        from TensorNAS.Layers.Conv2D import Args as conv_args

        return [
            Conv2D(
                input_shape=input_shape,
                parent_block=self,
                args={conv_args.FILTERS: 16, conv_args.PADDING: ArgPadding.SAME},
            )
        ]

    def generate_sub_block(self, input_shape, layer_type):
        from TensorNAS.Layers.Conv2D.DepthwiseConv2D import Layer as DepthwiseConv2D
        from TensorNAS.Layers.Conv2D.PointwiseConv2D import Layer as PointwiseConv2D
        from TensorNAS.Layers.Conv2D.GroupedPointwiseConv2D import (
            Layer as GroupedPointwise2D,
        )

        if layer_type == self.SubBlocks.GROUPEDCONV2D:
            return [GroupedPointwise2D(input_shape=input_shape, parent_block=self)]
        elif layer_type == self.SubBlocks.POINTWISECONV2D:
            return [PointwiseConv2D(input_shape=input_shape, parent_block=self)]
        elif layer_type == self.SubBlocks.DEPTHWISECONV2D:
            return [DepthwiseConv2D(input_shape=input_shape, parent_block=self)]

        return []

    def generate_constrained_output_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Conv2D.GroupedPointwiseConv2D import (
            Layer as GroupedPointwiseConv2D,
        )
        from TensorNAS.Layers.Shuffle import Layer as Shuffle
        from TensorNAS.Layers.Conv2D.DepthwiseConv2D import Layer as DepthwiseConv2D
        from TensorNAS.Core.Layer import ArgActivations, ArgPadding
        from TensorNAS.Layers.Conv2D import Args as conv_args

        residual_channel_depth = input_shape[-1]
        bottleneck_filters = residual_channel_depth // 4

        layers = []
        layers.append(
            GroupedPointwiseConv2D(
                input_shape=input_shape,
                parent_block=self,
                args={
                    conv_args.FILTERS: bottleneck_filters,
                    conv_args.ACTIVATION: ArgActivations.RELU,
                    conv_args.PADDING: ArgPadding.SAME,
                },
            )
        )
        layers.append(
            Shuffle(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
            )
        )
        layers.append(
            DepthwiseConv2D(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                args={
                    conv_args.KERNEL_SIZE: (3, 3),
                    conv_args.PADDING: ArgPadding.SAME,
                },
            )
        )
        layers.append(
            GroupedPointwiseConv2D(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                args={
                    conv_args.FILTERS: residual_channel_depth,
                    conv_args.ACTIVATION: ArgActivations.RELU,
                },
            )
        )
        return layers

    def get_keras_layers(self, input_tensor):
        from TensorNAS.Tools.TensorFlow import shortcut

        tmp = input_tensor
        for sb in self.input_blocks + self.middle_blocks + self.output_blocks:
            tmp = sb.get_keras_layers(tmp)
        return shortcut(input_tensor, tmp)
