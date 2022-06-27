from TensorNAS.Core.Block import Block
from enum import Enum, auto


class Block(Block):

    MAX_SUB_BLOCKS = 0
    MIN_SUB_BLOCKS = 0

    class SubBlocks(Enum):

        SEPARABLE_CONV_S1 = auto()
        SEPARABLE_CONV_S2 = auto()

    def generate_constrained_middle_sub_blocks(self, input_shape, args=None):
        from TensorNAS.Layers.Conv2D.SeparableConv2D import Layer as SeperableConv2D
        from TensorNAS.Layers.Conv2D.SeparableConv2D import Args as sep_conv_args
        from TensorNAS.Core.Layer import ArgPadding, ArgInitializers, ArgRegularizers

        filters = 8
        strides = (1, 1)

        if sep_conv_args.FILTERS in args:
            filters = args.get(sep_conv_args.FILTERS, 8)

        if sep_conv_args.STRIDES in args:
            strides = args.get(sep_conv_args.STRIDES, (1, 1))

        args = {
            sep_conv_args.FILTERS: filters,
            sep_conv_args.KERNEL_SIZE: (3, 3),
            sep_conv_args.PADDING: ArgPadding.SAME,
            sep_conv_args.STRIDES: strides,
            sep_conv_args.DEPTHWISE_INITIALIZER: ArgInitializers.HE_NORMAL,
            sep_conv_args.POINTWISE_INITIALIZER: ArgInitializers.HE_NORMAL,
            sep_conv_args.DEPTHWISE_REGULARIZER: (ArgRegularizers.L2, 1e-4),
            sep_conv_args.POINTWISE_REGULARIZER: (ArgRegularizers.L2, 1e-4),
        }

        return [
            SeperableConv2D(
                input_shape=input_shape,
                parent_block=self,
                args=args,
            )
        ]

    def generate_sub_block(self, input_shape, layer_type, args=None):
        from TensorNAS.Layers.Conv2D.SeparableConv2D import Layer as SeperableConv2D
        from TensorNAS.Layers.Conv2D.SeparableConv2D import Args as sep_conv_args
        from TensorNAS.Core.Layer import ArgPadding, ArgInitializers, ArgRegularizers

        filters = 8

        if sep_conv_args.FILTERS in args:
            filters = args.get(sep_conv_args.FILTERS, 8)

        args = {
            sep_conv_args.FILTERS: filters,
            sep_conv_args.KERNEL_SIZE: (3, 3),
            sep_conv_args.PADDING: ArgPadding.SAME,
            sep_conv_args.DEPTHWISE_INITIALIZER: ArgInitializers.HE_NORMAL,
            sep_conv_args.POINTWISE_INITIALIZER: ArgInitializers.HE_NORMAL,
            sep_conv_args.DEPTHWISE_REGULARIZER: (ArgRegularizers.L2, 1e-4),
            sep_conv_args.POINTWISE_REGULARIZER: (ArgRegularizers.L2, 1e-4),
        }

        if layer_type == self.SubBlocks.SEPARABLE_CONV_S1:
            args[sep_conv_args.STRIDES] = (1, 1)

            return [
                SeperableConv2D(
                    input_shape=input_shape,
                    parent_block=self,
                    args=args,
                )
            ]
        elif layer_type == self.SubBlocks.SEPARABLE_CONV_S2:
            args[sep_conv_args.STRIDES] = (2, 2)

            return [
                SeperableConv2D(
                    input_shape=input_shape,
                    parent_block=self,
                    args=args,
                )
            ]

        return []
