from TensorNAS.Core.Block import Block
from enum import Enum, auto


class Block(Block):
    """
    Block used for performing classification

    An optional class_count parameter specifies if there is a known number of output classes. This would be required
    if the classification block is the final block in a model, thus responsible for the NN output.

    If the classification block is not the output then is does not necessarily have a required number of outputs,
    meaning it can be a random number
    """

    DROPOUT_RATE_MAX = 0.2
    MAX_SUB_BLOCKS = 0
    MIN_SUB_BLOCKS = 0

    class SubBlocks(Enum):

        FLATTEN = auto()
        HIDDENDENSE = auto()
        DROPOUT = auto()

    def __init__(self, input_shape, parent_block, class_count):
        self.class_count = class_count

        super().__init__(input_shape, parent_block)

    def generate_constrained_output_sub_blocks(self, input_shape):

        from TensorNAS.Layers.Flatten import Layer as Flatten
        from TensorNAS.Layers.Dense.OutputDense import Layer as OutputDense
        from TensorNAS.Layers.Dense import Args as dense_args

        blocks = []

        blocks.append(Flatten(input_shape=input_shape, parent_block=self))
        blocks.append(
            OutputDense(
                input_shape=blocks[-1].get_output_shape(),
                parent_block=self,
                args={dense_args.UNITS: self.class_count},
            )
        )

        return blocks

    def generate_sub_block(self, input_shape, layer_type):
        from TensorNAS.Layers.Flatten import Layer as Flatten
        from TensorNAS.Layers.Dropout import Layer as Dropout
        from TensorNAS.Layers.Dense.HiddenDense import Layer as HiddenDense
        from TensorNAS.Layers.Dropout import Args as dropout_args

        if layer_type == self.SubBlocks.FLATTEN:
            return [
                Flatten(
                    input_shape=input_shape,
                    parent_block=self,
                )
            ]
        elif layer_type == self.SubBlocks.HIDDENDENSE:
            return [
                HiddenDense(
                    input_shape=input_shape,
                    parent_block=self,
                )
            ]
        elif layer_type == self.SubBlocks.DROPOUT:
            return [
                Dropout(
                    input_shape=input_shape,
                    parent_block=self,
                    args={dropout_args.RATE: self.DROPOUT_RATE_MAX},
                )
            ]
        return []
