from TensorNAS.Core.Block import Block
from enum import Enum, auto


class Block(Block):
    """
    Layers that can be used in the extraction of features
    """

    MAX_SUB_BLOCKS = 2

    class SubBlocks(Enum):

        CONV2D = auto()
        MAXPOOL2D = auto()

    def generate_constrained_middle_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Conv2D.Conv2D import Layer as Conv2D
        from TensorNAS.Layers.Pool.MaxPool2D import Layer as MaxPool2D

        blocks = []

        blocks.append(Conv2D(input_shape=input_shape, parent_block=self))
        blocks.append(
            MaxPool2D(input_shape=blocks[-1].get_output_shape(), parent_block=self)
        )

        return blocks

    def generate_random_sub_block(self, input_shape, layer_type):
        from TensorNAS.Layers.Conv2D.Conv2D import Layer as Conv2D
        from TensorNAS.Layers.Pool.MaxPool2D import Layer as MaxPool2D

        if layer_type == self.SubBlocks.CONV2D:
            return [Conv2D(input_shape=input_shape, parent_block=self)]
        elif layer_type == self.SubBlocks.MAXPOOL2D:
            return [MaxPool2D(input_shape=input_shape, parent_block=self)]
        return []
