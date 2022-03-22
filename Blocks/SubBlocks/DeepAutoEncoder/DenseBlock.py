from TensorNAS.Core.Block import Block
from enum import Enum, auto


class Block(Block):
    class SubBlocks(Enum):

        DENSE = auto()
        BATCH_NORMALIZATION = auto()
        ACTIVATION = auto()

    def __init__(self, input_shape, parent_block, units=128):
        self.units = units

        super().__init__(input_shape=input_shape, parent_block=parent_block)

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Dense import Args as dense_args
        from TensorNAS.Layers.Activation import Args as activation_args
        from TensorNAS.Core.Layer import ArgActivations
        from TensorNAS.Layers.Dense.HiddenDense import Layer as HiddenDenseLayer
        from TensorNAS.Layers.BatchNormalization import Layer as BatchNormalizationLayer
        from TensorNAS.Layers.Activation import Layer as ActivationLayer

        dense_args = {
            dense_args.UNITS: self.units,
            dense_args.ACTIVATION: ArgActivations.NONE,
        }
        activation_args = {activation_args.ACTIVATION: ArgActivations.RELU}

        layers = [
            HiddenDenseLayer(
                input_shape=input_shape, parent_block=self, args=dense_args
            )
        ]
        layers.append(
            BatchNormalizationLayer(
                input_shape=layers[-1].get_output_shape(), parent_block=self
            )
        )
        layers.append(
            ActivationLayer(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                args=activation_args,
            )
        )

        return layers
