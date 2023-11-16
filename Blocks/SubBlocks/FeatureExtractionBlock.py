from TensorNAS.Core.Block import Block
from enum import Enum, auto


class Block(Block):
    """
    Layers that can be used in the extraction of features
    """

    MAX_SUB_BLOCKS = 0
    MIN_SUB_BLOCKS = 0

    class SubBlocks(Enum):
        CONV2D = auto()
        DEPTHWISE_CONV2D = auto()
        POINTWISE_CONV2D = auto()
        AVERAGE_POOL_2D = auto()
        MAX_POOL2D = auto()

    def generate_constrained_middle_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Conv2D.Conv2D import Layer as Conv2D
        from TensorNAS.Layers.Conv2D import Args as conv2d_args
        from TensorNAS.Layers.Pool.MaxPool2D import Layer as MaxPool2D
        from TensorNAS.Layers.Pool import Args as pool_args
        from TensorNAS.Core.Layer import (
            ArgPadding,
            ArgActivations,
            ArgRegularizers,
            ArgInitializers,
        )

        blocks = []

        conv_args = {
            conv2d_args.FILTERS: 32,
            conv2d_args.KERNEL_SIZE: (3, 3),
            conv2d_args.DILATION_RATE: (1, 1),
            conv2d_args.STRIDES: (1, 1),
            conv2d_args.ACTIVATION: ArgActivations.NONE,
            conv2d_args.REGULARIZER: (ArgRegularizers.NONE, 0),
            conv2d_args.GROUPS: 1,
            conv2d_args.INITIALIZER: ArgInitializers.GLOROT_UNIFORM,
        }
        pool_args = {
            pool_args.POOL_SIZE: (2, 2),
            pool_args.STRIDES: None,
            pool_args.PADDING: ArgPadding.VALID,
        }

        blocks.append(
            Conv2D(input_shape=input_shape, parent_block=self, args=conv_args)
        )
        blocks.append(
            MaxPool2D(
                input_shape=blocks[-1].get_output_shape(),
                parent_block=self,
                args=pool_args,
            )
        )

        return blocks

    def generate_sub_block(self, input_shape, layer_type):
        from TensorNAS.Layers.Conv2D.Conv2D import Layer as Conv2D
        from TensorNAS.Layers.Conv2D.PointwiseConv2D import Layer as PointwiseConv2D
        from TensorNAS.Layers.Conv2D.DepthwiseConv2D import Layer as DepthwiseConv2D
        from TensorNAS.Layers.Pool.MaxPool2D import Layer as MaxPool2D
        from TensorNAS.Layers.Pool.AveragePooling2D import Layer as AvgPool2D

        if layer_type == self.SubBlocks.CONV2D:
            return [Conv2D(input_shape=input_shape, parent_block=self)]
        elif layer_type == self.SubBlocks.POINTWISE_CONV2D:
            return [PointwiseConv2D(input_shape=input_shape, parent_block=self)]
        elif layer_type == self.SubBlocks.DEPTHWISE_CONV2D:
            return [DepthwiseConv2D(input_shape=input_shape, parent_block=self)]
        elif layer_type == self.SubBlocks.MAX_POOL2D:
            return [MaxPool2D(input_shape=input_shape, parent_block=self)]
        elif layer_type == self.SubBlocks.AVERAGE_POOL_2D:
            return [AvgPool2D(input_shape=input_shape, parent_block=self)]

        return []
