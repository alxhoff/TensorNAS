from TensorNAS.Core.Block import Block
from enum import Enum, auto


class Block(Block):
    """
    Layers that can be used in the extraction of features
    """

    MAX_SUB_BLOCKS = 0
    MIN_SUB_BLOCKS = 0

    class SubBlocks(Enum):
        FLATTEN = auto()
        HIDDENDENSE = auto()

    def generate_constrained_middle_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Flatten import Layer as Flatten
        from TensorNAS.Layers.Dense.HiddenDense import Layer as HiddenDense
        from TensorNAS.Layers.Dense import Args as dense_args
        from TensorNAS.Core.Layer import ArgActivations

        args = {dense_args.UNITS: 128, dense_args.ACTIVATION: ArgActivations.RELU}

        blocks = []

        blocks.append(Flatten(input_shape=input_shape, parent_block=self))
        blocks.append(
            HiddenDense(
                input_shape=blocks[-1].get_output_shape(), parent_block=self, args=args
            )
        )

        return blocks

    def generate_sub_block(self, input_shape, layer_type):
        from TensorNAS.Layers.Dense.HiddenDense import Layer as HiddenDense
        from TensorNAS.Layers.Dense import Args as dense_args
        from TensorNAS.Core.Layer import ArgActivations

        args = {dense_args.UNITS: 128, dense_args.ACTIVATION: ArgActivations.RELU}

        return [HiddenDense(input_shape=input_shape, parent_block=self, args=args)]
