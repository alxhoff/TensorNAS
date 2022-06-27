from TensorNAS.Core.Block import Block
from enum import Enum, auto


class Block(Block):
    class SubBlocks(Enum):

        AVERAGE_POOL_2D = auto()
        FLATTEN = auto()
        OUTPUT_DENSE = auto()

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Pool.AveragePooling2D import Layer as AveragePool2D
        from TensorNAS.Layers.Flatten import Layer as Flatten
        from TensorNAS.Layers.Dense.OutputDense import Layer as OutputDense
        from TensorNAS.Layers.Pool import Args as pool_args
        from TensorNAS.Layers.Dense import Args as dense_args
        from TensorNAS.Core.Layer import ArgActivations, ArgInitializers, ArgPadding

        import numpy as np

        layers = []

        pool_size = int(np.amin(input_shape))
        pool_size = (pool_size, pool_size)

        layers.append(
            AveragePool2D(
                input_shape=input_shape,
                parent_block=self,
                args={
                    pool_args.POOL_SIZE: pool_size,
                    pool_args.STRIDES: pool_size,
                    pool_args.PADDING: ArgPadding.SAME,
                },
            )
        )

        layers.append(
            Flatten(input_shape=layers[-1].get_output_shape(), parent_block=self)
        )

        layers.append(
            OutputDense(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                args={
                    dense_args.UNITS: self.get_block_architecture().class_count,
                    dense_args.ACTIVATION: ArgActivations.SOFTMAX,
                    dense_args.INITIALIZER: ArgInitializers.HE_NORMAL,
                },
            )
        )

        return layers

    def generate_sub_block(self, input_shape, layer_type):

        if layer_type == self.SubBlocks.FLATTEN:
            from TensorNAS.Layers.Flatten import Layer as Flatten

            return [Flatten(input_shape=input_shape, parent_block=self)]

        elif layer_type == self.SubBlocks.OUTPUT_DENSE:
            from TensorNAS.Layers.Dense.OutputDense import Layer as OutputDense
            from TensorNAS.Layers.Dense import Args as dense_args
            from TensorNAS.Core.Layer import ArgInitializers, ArgActivations

            return [
                OutputDense(
                    input_shape=input_shape,
                    parent_block=self,
                    args={
                        dense_args.UNITS: self.get_block_architecture().class_count,
                        dense_args.ACTIVATION: ArgActivations.SOFTMAX,
                        dense_args.INITIALIZER: ArgInitializers.HE_NORMAL,
                    },
                )
            ]

        elif layer_type == self.SubBlocks.AVERAGE_POOL_2D:
            from TensorNAS.Layers.Pool.AveragePooling2D import Layer as AveragePool2D
            from TensorNAS.Layers.Pool import Args as pool_args
            import numpy as np

            pool_size = int(np.amin(input_shape))
            pool_size = (pool_size, pool_size)

            return [
                AveragePool2D(
                    input_shape=input_shape,
                    parent_block=self,
                    args={pool_args.POOL_SIZE: pool_size},
                )
            ]

        return []
