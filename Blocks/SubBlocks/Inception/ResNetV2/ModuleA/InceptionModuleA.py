from TensorNAS.Tools.TensorFlow.ParallelMidBlock import ParallelMidBlock
from enum import Enum, auto


class Block(ParallelMidBlock):
    class SubBlocks(Enum):

        POINTWISE_CONV2D = auto()
        MODULE_A_BRANCH_1 = auto()
        MODULE_A_BRANCH_2 = auto()

    def generate_constrained_output_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Conv2D import Args as conv_args

        from TensorNAS.Layers.Conv2D.PointwiseConv2D import Layer as PointwiseConv2D

        return [
            PointwiseConv2D(
                input_shape=input_shape,
                parent_block=self,
                args={conv_args.FILTERS: 384},
            )
        ]

    def generate_constrained_middle_sub_blocks(self, input_shape):

        from TensorNAS.Layers.Conv2D import Args as conv_args

        from TensorNAS.Layers.Conv2D.PointwiseConv2D import Layer as PointwiseConv2D
        from TensorNAS.Blocks.SubBlocks.Inception.InceptionModuleABranch1 import (
            Block as MAB1,
        )
        from TensorNAS.Blocks.SubBlocks.Inception.InceptionModuleABranch2 import (
            Block as MAB2,
        )

        layers = []
        layers.append(
            PointwiseConv2D(
                input_shape=input_shape, parent_block=self, args={conv_args.FILTERS: 32}
            )
        )
        layers.append(MAB1(input_shape=input_shape, parent_block=self))
        layers.append(MAB2(input_shape=input_shape, parent_block=self))

        return layers

    def generate_constrained_input_sub_blocks(self, input_shape):

        from TensorNAS.Layers.Activation import Layer as Activation
        from TensorNAS.Layers.Activation import Args as activation_args
        from TensorNAS.Core.Layer import ArgActivations

        return [
            Activation(
                input_shape=input_shape,
                parent_block=self,
                args={activation_args.ACTIVATION: ArgActivations.RELU},
            )
        ]

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
