from tensornas.mutator import *
from tensornas.block import Block
from tensornas.layerargsbuilder import layerargsbuilder
from tensornas import layers


class FeatureExtractionBlockLayerTypes(Enum):
    """
    Layers that can be used in the generation of a feature extraction block are enumerated here for random selection
    """

    CONV2D = auto()
    MAXPOOL2D = auto()


class FeatureExtractionBlock(Block):
    """
    Layers that can be used in the extraction of features
    """

    MAX_SUB_BLOCKS = 10
    SUB_BLOCK_TYPES = FeatureExtractionBlockLayerTypes

    def validate(self):
        return True

    def generate_constrained_input_sub_blocks(self):
        pass

    def generate_random_sub_block(self, layer_type):
        if layer_type == self.SUB_BLOCK_TYPES.CONV2D.value:
            return layerargsbuilder.generateconvolutionallayer()
        elif layer_type == self.SUB_BLOCK_TYPES.MAXPOOL2D.value:
            return layerargsbuilder.generatepoolinglayer()


class ClassificationBlockLayerTypes(Enum):
    """
    Layers that can be used in the generation of a feature extraction block are enumerated here for random selection
    """

    FLATTEN = auto()
    DENSE = auto()
    DROPOUT = auto()


class ClassificationBlock(Block):
    """
    Block used for performing classification

    An optional class_count parameter specifies if there is a known number of output classes. This would be required
    if the classification block is the final block in a model, thus responsible for the NN output.

    If the classification block is not the output then is does not necessarily have a required number of outputs,
    meaning it can be a random number
    """

    MAX_SUB_BLOCKS = 10
    SUB_BLOCK_TYPES = ClassificationBlockLayerTypes

    def __init__(self, input_shape, parent_block, class_count=None):
        self.class_count = class_count

        super().__init__(input_shape, parent_block)

    def validate(self):
        ret = True
        if not isinstance(self.sub_blocks[-1], layers.OutputDenseLayer):
            ret = False
        return ret

    def generate_constrained_input_sub_blocks(self):
        self.sub_blocks.append(
            layerargsbuilder.generateoutputdenselayer(self.class_count)
        )

    def generate_random_sub_block(self, layer_type):
        if layer_type == self.SUB_BLOCK_TYPES.FLATTEN.value:
            return layerargsbuilder.generateflattenlayer()
        elif layer_type == self.SUB_BLOCK_TYPES.DENSE.value:
            return layerargsbuilder.generatedenselayer()
        elif layer_type == self.SUB_BLOCK_TYPES.DROPOUT.value:
            return layerargsbuilder.generatedropoutlayer()
