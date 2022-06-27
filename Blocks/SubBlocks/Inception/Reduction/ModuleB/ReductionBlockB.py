from TensorNAS.Tools.TensorFlow.ParallelMidBlock import ParallelMidBlock
from enum import Enum, auto


class Block(ParallelMidBlock):
    class SubBlocks(Enum):

        POINTWISE_CONV2D = auto()
        MODULE_A_BRANCH_1 = auto()
        MODULE_A_BRANCH_2 = auto()

    def generate_constrained_middle_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Pool.MaxPool2D import Layer as MaxPool2D
        from TensorNAS.Blocks.SubBlocks.Inception.ReductionBlockBBranch1 import (
            Block as RBB1,
        )
        from TensorNAS.Blocks.SubBlocks.Inception.ReductionBlockBBranch2 import (
            Block as RBB2,
        )
        from TensorNAS.Layers.Pool import Args as pool_args
        from TensorNAS.Core.Layer import ArgPadding

        layers = []
        layers.append(
            MaxPool2D(
                input_shape=input_shape,
                parent_block=self,
                args={
                    pool_args.POOL_SIZE: (3, 3),
                    pool_args.STRIDES: (2, 2),
                    pool_args.PADDING: ArgPadding.VALID,
                },
            )
        )
        layers.append(RBB1(input_shape=input_shape, parent_block=self))
        layers.append(RBB1(input_shape=input_shape, parent_block=self))
        layers.append(RBB2(input_shape=input_shape, parent_block=self))

        return layers

    def generate_sub_block(self, input_shape, layer_type):

        if layer_type == self.SubBlocks.POINTWISE_CONV2D:
            from TensorNAS.Layers.Conv2D.PointwiseConv2D import Layer as PointwiseConv2D

            return [PointwiseConv2D(input_shape=input_shape, parent_block=self)]
        elif layer_type == self.SubBlocks.MODULE_A_BRANCH_1:
            from TensorNAS.Blocks.SubBlocks.Inception.InceptionModuleABranch1 import (
                Block as MAB1,
            )

            return [MAB1(input_shape=input_shape, parent_block=self)]
        elif layer_type == self.SubBlocks.MODULE_A_BRANCH_2:
            from TensorNAS.Blocks.SubBlocks.Inception.InceptionModuleABranch2 import (
                Block as MAB2,
            )

            return [MAB2(input_shape=input_shape, parent_block=self)]

        return []
