from TensorNAS.Core.Block import Block


class Block(Block):

    MAX_SUB_BLOCKS = 1

    from enum import Enum

    class SubBlocks(Enum):
        from enum import auto

        DEPTHWISE_CONV2D = auto()
        BATCH_NORMALIZATION_AND_ACTIVATION = auto()

    def generate_random_sub_block(self, input_shape, block_type):
        from TensorNAS.Layers.Conv2D.Conv2D import Layer as Conv2D
        from TensorNAS.Layers.BatchNormalization import Layer as BatchNormalization
        from TensorNAS.Layers.Activation import Layer as Activation
        from TensorNAS.Layers.Dropout import Layer as Dropout
        from TensorNAS.Layers.Conv2D import Args as conv2d_args
        from TensorNAS.Layers.Activation import Args as activation_args
        from TensorNAS.Layers.Dropout import Args as dropout_args
        from TensorNAS.Core.Layer import ArgRegularizers
        from TensorNAS.Core.Layer import ArgPadding
        from TensorNAS.Core.Layer import ArgActivations

        layers = []

        layers.append(
            Conv2D(
                input_shape=input_shape,
                parent_block=self,
                args={
                    conv2d_args.FILTERS: 64,
                    conv2d_args.KERNEL_SIZE: (10, 4),
                    conv2d_args.STRIDES: (2, 2),
                    conv2d_args.KERNEL_REGULARIZER: (ArgRegularizers.L2, 1e-4),
                    conv2d_args.PADDING: ArgPadding.SAME,
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
        layers.append(
            Dropout(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                args={dropout_args.RATE: 0.2},
            )
        )

        return layers
