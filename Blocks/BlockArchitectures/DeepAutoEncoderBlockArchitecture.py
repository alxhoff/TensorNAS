from TensorNAS.Core.BlockArchitecture import AreaUnderCurveBlockArchitecture
from enum import Enum, auto


class Block(AreaUnderCurveBlockArchitecture):
    class SubBlocks(Enum):

        AUTO_ENCODER_BLOCK = auto()

    def generate_random_sub_block(self, input_shape, subblock_type):
        from TensorNAS.Blocks.SubBlocks.DeepAutoEncoder.DenseBlockArray import (
            Block as DAEArrayBlock,
        )

        return [DAEArrayBlock(input_shape=input_shape, parent_block=self)]

    def generate_constrained_output_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Dense import Layer as Dense
        from TensorNAS.Layers.Dense import Args as dense_args
        from TensorNAS.Core.Layer import ArgActivations

        return [
            Dense(
                input_shape=input_shape,
                parent_block=self,
                args={
                    dense_args.UNITS: int(self.get_input_shape()[0]),
                    dense_args.ACTIVATION: ArgActivations.NONE,
                },
            )
        ]
