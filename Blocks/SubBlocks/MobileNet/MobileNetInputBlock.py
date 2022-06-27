from TensorNAS.Core.Block import Block
from enum import Enum, auto


class Block(Block):

    MAX_SUB_BLOCKS = 0
    MIN_SUB_BLOCKS = 0

    class SubBlocks(Enum):

        CONV2D = auto()
        BATCH_NORMALIZATION = auto()
        ACTIVATION = auto()

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Conv2D.Conv2D import Layer as Conv2D
        from TensorNAS.Layers.Conv2D import Args as conv_args
        from TensorNAS.Layers.BatchNormalization import Layer as BatchNormalization
        from TensorNAS.Layers.Activation import Layer as Activation
        from TensorNAS.Layers.Activation import Args as activation_args
        from TensorNAS.Core.Layer import (
            ArgPadding,
            ArgRegularizers,
            ArgInitializers,
            ArgActivations,
        )

        args = {
            conv_args.FILTERS: 8,
            conv_args.KERNEL_SIZE: (3, 3),
            conv_args.PADDING: ArgPadding.SAME,
            conv_args.STRIDES: (2, 2),
            conv_args.REGULARIZER: (ArgRegularizers.L2, 1e-4),
            conv_args.INITIALIZER: ArgInitializers.HE_NORMAL,
        }

        layers = []
        layers.append(Conv2D(input_shape=input_shape, parent_block=self, args=args))
        layers.append(
            BatchNormalization(
                input_shape=layers[-1].get_output_shape(), parent_block=self
            )
        )
        layers.append(
            Activation(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                args={activation_args.ACTIVATION: ArgActivations.RELU},
            )
        )

        return layers

    def generate_sub_block(self, input_shape, layer_type, args=None):

        if layer_type == self.SubBlocks.CONV2D:
            from TensorNAS.Layers.Conv2D.Conv2D import Layer as Conv2D
            from TensorNAS.Layers.Conv2D import Args as conv_args
            from TensorNAS.Core.Layer import (
                ArgPadding,
                ArgRegularizers,
                ArgInitializers,
            )

            args = {
                conv_args.FILTERS: 8,
                conv_args.KERNEL_SIZE: (3, 3),
                conv_args.PADDING: ArgPadding.SAME,
                conv_args.STRIDES: (2, 2),
                conv_args.REGULARIZER: (ArgRegularizers.L2, 1e-4),
                conv_args.INITIALIZER: ArgInitializers.HE_NORMAL,
            }
            return [Conv2D(input_shape=input_shape, parent_block=self, args=args)]
        elif layer_type == self.SubBlocks.ACTIVATION:
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
        elif layer_type == self.SubBlocks.BATCH_NORMALIZATION:
            from TensorNAS.Layers.BatchNormalization import Layer as BatchNormalization

            return [BatchNormalization(input_shape=input_shape, parent_block=self)]

        return []
