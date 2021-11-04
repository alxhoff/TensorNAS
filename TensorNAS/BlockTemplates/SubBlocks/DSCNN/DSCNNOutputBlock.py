from enum import Enum, auto

from TensorNAS.Core.Block import Block


class DSCNNConvBlockLayerTypes(Enum):

    DEPTHWISE_CONV2D = auto()
    BATCH_NORMALIZATION_AND_ACTIVATION = auto()


class Block(Block):

    MAX_SUB_BLOCKS = 1
    SUB_BLOCK_TYPES = DSCNNConvBlockLayerTypes

    def __init__(self, input_shape, parent_block, class_count):
        self.class_count = class_count

        super().__init__(input_shape, parent_block)

    def generate_constrained_output_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Dropout import Layer as Dropout
        from TensorNAS.Layers.Pool.AveragePooling2D import Layer as AveragePool2D
        from TensorNAS.Layers.Flatten import Layer as Flatten
        from TensorNAS.Layers.Dense.OutputDense import Layer as OutputDense
        from TensorNAS.Layers.Dense import Args as dense_args
        from TensorNAS.Core.Layer import ArgActivations

        layers = []

        layers.append(
            Dropout(
                input_shape=input_shape,
                parent_block=self,
            )
        )
        layers.append(
            AveragePool2D(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
            )
        )
        layers.append(
            Flatten(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
            )
        )
        layers.append(
            OutputDense(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                args={
                    dense_args.UNITS: self.class_count,
                    dense_args.ACTIVATION: ArgActivations.SOFTMAX,
                },
            )
        )

        return layers
