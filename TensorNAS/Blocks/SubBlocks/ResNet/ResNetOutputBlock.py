from TensorNAS.Core.Block import Block


class Block(Block):

    from enum import Enum

    class SubBlocks(Enum):
        from enum import auto

        NONE = auto()

    def generate_constrained_input_sub_blocks(self, input_shape):
        from TensorNAS.Layers.Pool.AveragePooling2D import Layer as AveragePool2D
        from TensorNAS.Layers.Flatten import Layer as Flatten
        from TensorNAS.Layers.Dense.OutputDense import Layer as OutputDense
        from TensorNAS.Layers.Pool import Args as pool_args
        from TensorNAS.Layers.Dense import Args as dense_args
        import numpy as np

        layers = []

        pool_size = np.amin(input_shape)
        pool_size = (pool_size, pool_size)

        layers.append(
            AveragePool2D(
                input_shape=input_shape,
                parent_block=self,
                args={pool_args.POOL_SIZE: pool_size},
            )
        )

        layers.append(
            Flatten(input_shape=layers[-1].get_output_shape(), parent_block=self)
        )

        layers.append(
            OutputDense(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                args={dense_args.UNITS: self.get_block_architecture().class_count},
            )
        )

        return layers
