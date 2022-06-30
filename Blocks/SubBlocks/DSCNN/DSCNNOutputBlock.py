from TensorNAS.Core.Block import Block
from enum import Enum, auto


class Block(Block):

    MAX_SUB_BLOCKS = 0

    class SubBlocks(Enum):

        DEPTHWISE_CONV2D = auto()
        BATCH_NORMALIZATION_AND_ACTIVATION = auto()

    def generate_constrained_output_sub_blocks(self, input_shape, args):
        from TensorNAS.Layers.Dropout import Layer as Dropout
        from TensorNAS.Layers.Pool.AveragePooling2D import Layer as AveragePool2D
        from TensorNAS.Layers.Flatten import Layer as Flatten
        from TensorNAS.Layers.Dense.OutputDense import Layer as OutputDense
        from TensorNAS.Layers.Dense import Args as dense_args
        from TensorNAS.Layers.Pool import Args as pool_args
        from TensorNAS.Core.Layer import ArgActivations, ArgPadding

        layers = []

        pool_size = args.get(pool_args.POOL_SIZE, (1, 1))

        # layers.append(
        #     Dropout(
        #         input_shape=input_shape,
        #         parent_block=self,
        #     )
        # )
        # layers.append(
        #     AveragePool2D(
        #         input_shape=layers[-1].get_output_shape(),
        #         parent_block=self,
        #         args = {pool_args.POOL_SIZE: (layers[-1].get_output_shape()[0]/2, layers[-1].get_output_shape()[1]/2)}
        #     )
        # )
        layers.append(
            AveragePool2D(
                input_shape=input_shape,
                parent_block=self,
                args={
                    pool_args.POOL_SIZE: pool_size,
                    pool_args.STRIDES: pool_size,
                    pool_args.PADDING: ArgPadding.VALID,
                },
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
                    dense_args.UNITS: self.get_block_architecture().class_count,
                    dense_args.ACTIVATION: ArgActivations.SOFTMAX,
                },
            )
        )

        return layers
