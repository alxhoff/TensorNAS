from TensorNAS.Core.Block import Block
from enum import Enum, auto


class Block(Block):
    class SubBlocks(Enum):

        CONV2D = auto()
        POINTWISE_CONV2D = auto()

    def generate_constrained_middle_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Conv2D.Conv2D import Layer as Conv2D
        from TensorNAS.Layers.Conv2D.PointwiseConv2D import Layer as PointwiseConv2D
        from TensorNAS.Layers.Conv2D import Args as conv_args

        layers = []
        layers.append(
            PointwiseConv2D(
                input_shape=input_shape,
                parent_block=self,
                args={conv_args.FILTERS: 128},
            )
        )
        layers.append(
            Conv2D(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                args={conv_args.KERNEL_SIZE: (1, 7), conv_args.FILTERS: 160},
            )
        )
        layers.append(
            Conv2D(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                args={conv_args.KERNEL_SIZE: (7, 1), conv_args.FILTERS: 192},
            )
        )

        return layers

    def generate_sub_block(self, input_shape, layer_type):

        if layer_type == self.SubBlocks.CONV2D:
            from TensorNAS.Layers.Conv2D.Conv2D import Layer as Conv2D
            from TensorNAS.Layers.Conv2D import Args as conv_args

            layers = [
                Conv2D(
                    input_shape=input_shape,
                    parent_block=self,
                    args={conv_args.KERNEL_SIZE: (7, 1)},
                )
            ]
            layers.append(
                Conv2D(
                    input_shape=layers[-1].get_output_shape(),
                    parent_block=self,
                    args={conv_args.KERNEL_SIZE: (1, 7)},
                )
            )

            return layers
        elif layer_type == self.SubBlocks.POINTWISE_CONV2D:
            from TensorNAS.Layers.Conv2D.PointwiseConv2D import Layer as PointwiseConv2D

            return [PointwiseConv2D(input_shape=input_shape, parent_block=self)]

        return []
