from TensorNAS.Tools.TensorFlow.ParallelMidBlock import ParallelMidBlock
from enum import Enum, auto


class Block(ParallelMidBlock):
    class SubBlocks(Enum):

        POINTWISE_CONV2D = auto()
        MODULE_B_BRANCH_1 = auto()

    def generate_constrained_output_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Conv2D import Args as conv_args
        from TensorNAS.Layers.Conv2D.PointwiseConv2D import Layer as PointwiseConv2D

        return [
            PointwiseConv2D(
                input_shape=input_shape,
                parent_block=self,
                args={conv_args.FILTERS: 1154},
            )
        ]

    def generate_constrained_middle_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Conv2D import Args as conv_args
        from TensorNAS.Layers.Conv2D.PointwiseConv2D import Layer as PointwiseConv2D
        from TensorNAS.Blocks.SubBlocks.Inception.InceptionModuleBBranch1 import (
            Block as MBB1,
        )

        layers = []
        layers.append(
            PointwiseConv2D(
                input_shape=input_shape,
                parent_block=self,
                args={conv_args.FILTERS: 192},
            )
        )
        layers.append(MBB1(input_shape=input_shape, parent_block=self))

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
        elif layer_type == self.SubBlocks.MODULE_B_BRANCH_1:
            from TensorNAS.Blocks.SubBlocks.Inception.InceptionModuleBBranch1 import (
                Block as MBB1,
            )

            return [MBB1(input_shape=input_shape, parent_block=self)]

        return []
