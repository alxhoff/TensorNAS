from tensornasmutator import *
from nasblock import Block
from tensornasbuilder import ModelBuilder
import tensornaslayers


class TopLevelBlockTypes(Enum):
    CLASSIFICATION_BLOCK = auto()
    FEATURE_EXTRACTION_BLOCK = auto()


class BlockArchitecture(Block):
    """
    Model architecutre that contains a collection of blocks

    This class represents in a block an entire model, for example, a feature extraction block that would be some sort of
    combination of convolutional layer(s) and maybe pooling. Another example of a block would be a classification block
    which could be flattening followed by a dense layer.
    """

    MAX_SUB_BLOCKS = 5
    SUB_BLOCK_TYPES = TopLevelBlockTypes

    def validate(self):
        ret = True
        if not isinstance(self.sub_blocks[-1], ClassificationBlock):
            ret = False
        return ret

    def generate_constrained_sub_blocks(self):
        self.sub_blocks.append(ClassificationBlock(self.input_shape, self.class_count))

    def get_random_sub_block(self, layer_type):
        if layer_type == self.SUB_BLOCK_TYPES.CLASSIFICATION_BLOCK.value:
            return ClassificationBlock(self.input_shape)
        elif layer_type == self.SUB_BLOCK_TYPES.FEATURE_EXTRACTION_BLOCK.value:
            return FeatureExtractionBlock(self.input_shape)

    def __init__(self, input_shape, class_count):
        self.class_count = class_count

        super().__init__(input_shape)


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

    def generate_constrained_sub_blocks(self):
        pass

    def get_random_sub_block(self, layer_type):
        if layer_type == self.SUB_BLOCK_TYPES.CONV2D:
            return ModelBuilder.generateconvolutionallayer()
        elif layer_type == self.SUB_BLOCK_TYPES.MAXPOOL2D:
            return ModelBuilder.generatepoolinglayer()


class ClassificationBlockLayerTypes(Enum):
    """
    Layers that can be used in the generation of a feature extraction block are enumberated here for random selection
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

    def __init__(self, input_shape, class_count=None):
        self.class_count = class_count

        super().__init__(input_shape)

    def validate(self):
        ret = True
        if not isinstance(self.sub_blocks[-1], tensornaslayers.OutputDenseLayer):
            ret = False
        return ret

    def generate_constrained_sub_blocks(self):
        self.sub_blocks.append(ModelBuilder.generateoutputdenselayer(self.class_count))

    def get_random_sub_block(self, layer_type):
        if layer_type == self.SUB_BLOCK_TYPES.FLATTEN.value:
            return ModelBuilder.generateflattenlayer()
        elif layer_type == self.SUB_BLOCK_TYPES.DENSE.value:
            return ModelBuilder.generatedenselayer()
        elif layer_type == self.SUB_BLOCK_TYPES.DROPOUT.value:
            return ModelBuilder.generatedropoutlayer()
