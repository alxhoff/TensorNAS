from TensorNAS.Core.BlockArchitecture import ClassificationBlockArchitecture
from enum import Enum, auto

from TensorNAS.Blocks.SubBlocks.Inception.InceptionModuleA import Block as InceptionA
from TensorNAS.Blocks.SubBlocks.Inception.InceptionModuleB import (
    Block as InceptionB,
)
from TensorNAS.Blocks.SubBlocks.Inception.InceptionModuleC import (
    Block as InceptionC,
)
from TensorNAS.Blocks.SubBlocks.Inception.ReductionBlockA import (
    Block as ReductionA,
)
from TensorNAS.Blocks.SubBlocks.Inception.ReductionBlockB import (
    Block as ReductionB,
)
from TensorNAS.Blocks.SubBlocks.TwoDClassificationBlock import (
    Block as TwoDClassification,
)

from TensorNAS.Layers.Dropout import Layer as Dropout
from TensorNAS.Layers.Pool.AveragePooling2D import Layer as AveragePool


class Block(ClassificationBlockArchitecture):

    MAX_SUB_BLOCKS = 3

    class SubBlocks(Enum):

        DROPOUT = auto()
        AVERAGE_POOL = auto()
        INCEPTION_A = auto()
        INCEPTION_B = auto()
        INCEPTION_C = auto()
        REDUCTION_A = auto()
        REDUCTION_B = auto()

    def generate_sub_block(self, input_shape, layer_type):

        if layer_type == self.SubBlocks.INCEPTION_A:
            return [InceptionA(input_shape=input_shape, parent_block=self)]
        elif layer_type == self.SubBlocks.INCEPTION_B:
            return [InceptionB(input_shape=input_shape, parent_block=self)]
        elif layer_type == self.SubBlocks.INCEPTION_C:
            return [InceptionC(input_shape=input_shape, parent_block=self)]
        elif layer_type == self.SubBlocks.REDUCTION_A:
            return [ReductionA(input_shape=input_shape, parent_block=self)]
        elif layer_type == self.SubBlocks.REDUCTION_B:
            return [ReductionB(input_shape=input_shape, parent_block=self)]
        elif layer_type == self.SubBlocks.DROPOUT:
            return [Dropout(input_shape=input_shape, parent_block=self)]
        elif layer_type == self.SubBlocks.AVERAGE_POOL:
            return [AveragePool(input_shape=input_shape, parent_block=self)]

    def generate_constrained_middle_sub_blocks(self, input_shape, args=None):

        layers = [InceptionA(input_shape=input_shape, parent_block=self)]
        layers.append(
            ReductionA(input_shape=layers[-1].get_output_shape(), parent_block=self)
        )
        layers.append(
            InceptionB(input_shape=layers[-1].get_output_shape(), parent_block=self)
        )
        layers.append(
            ReductionB(input_shape=layers[-1].get_output_shape(), parent_block=self)
        )
        layers.append(
            InceptionC(input_shape=layers[-1].get_output_shape(), parent_block=self)
        )
        layers.append(
            AveragePool(input_shape=layers[-1].get_output_shape(), parent_block=self)
        )
        layers.append(
            Dropout(input_shape=layers[-1].get_output_shape(), parent_block=self)
        )
        layers.append(
            TwoDClassification(
                input_shape=layers[-1].get_output_shape(),
                parent_block=self,
                class_count=self.class_count,
            )
        )

        return layers
