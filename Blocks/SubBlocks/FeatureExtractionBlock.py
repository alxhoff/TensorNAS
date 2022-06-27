from TensorNAS.Core.Block import Block
from enum import Enum, auto


class Block(Block):
    """
    Layers that can be used in the extraction of features
    """

    MAX_SUB_BLOCKS = 6
    MIN_SUB_BLOCKS = 2

    class SubBlocks(Enum):

        CONV2D = auto()
        DEPTHWISE_CONV2D = auto()
        POINTWISE_CONV2D = auto()
        AVERAGE_POOL_2D = auto()
        MAX_POOL2D = auto()

    def generate_constrained_middle_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Conv2D.Conv2D import Layer as Conv2D
        from TensorNAS.Layers.Pool.MaxPool2D import Layer as MaxPool2D

        blocks = []

        blocks.append(Conv2D(input_shape=input_shape, parent_block=self))
        blocks.append(
            MaxPool2D(input_shape=blocks[-1].get_output_shape(), parent_block=self)
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
