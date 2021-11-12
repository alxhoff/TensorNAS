from TensorNAS.Core.Block import Block


class Block(Block):
    """
    Layers that can be used in the extraction of features
    """

    MAX_SUB_BLOCKS = 2

    from enum import Enum

    class SubBlocks(Enum):
        from enum import auto

        CONV2D = auto()
        MAXPOOL2D = auto()

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Conv2D.Conv2D import Layer as Conv2D

        return [Conv2D(input_shape=input_shape, parent_block=self)]

    def generate_random_sub_block(self, input_shape, layer_type):
        from TensorNAS.Layers.Conv2D.Conv2D import Layer as Conv2D
        from TensorNAS.Layers.Pool.MaxPool2D import Layer as MaxPool2D

        if layer_type == self.SubBlocks.CONV2D:
            return [Conv2D(input_shape=input_shape, parent_block=self)]
        elif layer_type == self.SubBlocks.MAXPOOL2D:
            return [MaxPool2D(input_shape=input_shape, parent_block=self)]
        return []
