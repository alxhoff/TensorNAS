from TensorNAS.Core.Block import Block


class Block(Block):

    MAX_SUB_BLOCKS = 1

    from enum import Enum

    class SubBlocks(Enum):
        from enum import auto

        BATCH_NORMILIZATION = auto()
        ACTIVATION = auto()

    def generate_constrained_output_sub_blocks(self, input_shape):
        from TensorNAS.Layers.BatchNormalization import Layer as BatchNormalization
        from TensorNAS.Layers.Activation import Layer as Activation
        from TensorNAS.Layers.Activation import Args as activation_args
        from TensorNAS.Core.Layer import ArgActivations

        layers = []

        layers.append(
            BatchNormalization(
                input_shape=input_shape,
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
