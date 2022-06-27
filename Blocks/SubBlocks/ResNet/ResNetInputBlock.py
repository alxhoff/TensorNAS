from TensorNAS.Core.Block import Block
from enum import Enum, auto


class Block(Block):
    class SubBlocks(Enum):

        NONE = auto()

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Conv2D.Conv2D import Layer as Conv2D
        from TensorNAS.Layers.BatchNormalization import Layer as BatchNormalization
        from TensorNAS.Layers.Activation import Layer as Activation
        from TensorNAS.Layers.Conv2D import Args as conv2d_args
        from TensorNAS.Layers.Activation import Args as activation_args
        from TensorNAS.Core.Layer import (
            ArgPadding,
            ArgRegularizers,
            ArgActivations,
            ArgInitializers,
        )

        layers = []

        layers.append(
            Conv2D(
                input_shape=input_shape,
                parent_block=self,
                args={
                    conv2d_args.FILTERS: 16,
                    conv2d_args.KERNEL_SIZE: (3, 3),
                    conv2d_args.STRIDES: (1, 1),
                    conv2d_args.PADDING: ArgPadding.SAME,
                    conv2d_args.REGULARIZER: (ArgRegularizers.L2, 1e-4),
                    conv2d_args.ACTIVATION: ArgActivations.NONE,
                    conv2d_args.INITIALIZER: ArgInitializers.HE_NORMAL,
                },
            )
        )
        layers.append(
            BatchNormalization(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
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

    def generate_sub_block(self, input_shape, layer_type):
        from TensorNAS.Layers.Activation import Layer as Activation

        if layer_type == self.SubBlocks.CONV2D:
            from TensorNAS.Layers.Conv2D.Conv2D import Layer as Conv2D
            from TensorNAS.Layers.Conv2D import Args as conv2d_args
            from TensorNAS.Core.Layer import (
                ArgPadding,
                ArgRegularizers,
                ArgActivations,
                ArgInitializers,
            )

            return [
                Conv2D(
                    input_shape=input_shape,
                    parent_block=self,
                    args={
                        conv2d_args.FILTERS: 16,
                        conv2d_args.KERNEL_SIZE: (3, 3),
                        conv2d_args.STRIDES: (1, 1),
                        conv2d_args.PADDING: ArgPadding.SAME,
                        conv2d_args.REGULARIZER: (ArgRegularizers.L2, 1e-4),
                        conv2d_args.ACTIVATION: ArgActivations.NONE,
                        conv2d_args.INITIALIZER: ArgInitializers.HE_NORMAL,
                    },
                )
            ]
        elif layer_type == self.SubBlocks.BATCH_NORMALIZATION:
            from TensorNAS.Layers.BatchNormalization import Layer as BatchNormalization

            return [BatchNormalization(input_shape=input_shape, parent_block=self)]
        elif layer_type == self.SubBlocks.ACTIVATION:
            from TensorNAS.Layers.Activation import Args as activation_args
            from TensorNAS.Core.Layer import ArgActivations

            return [
                Activation(
                    input_shape=input_shape,
                    parent_block=self,
                    args={activation_args.ACTIVATION: ArgActivations.RELU},
                )
            ]

        return []
