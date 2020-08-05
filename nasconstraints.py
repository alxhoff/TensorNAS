import random
from tensornas import *
import itertools
# This class should generate models needed for creating individuals which is done using iterators and DEAP's initIterate
# The model builder should make sure that input and output sizes match and perform similar checks to make sure that
# layer order makes sense, ie. do not put a pooling layer before convoloutional layers etc.
# Individual layers should have checking built into their classes, eg. directly into Conv2DLayer, such that checking
# a model at each layer is as simple as looping through layers and running layer.validate(), breaking on False.

# This class should also return iterators that are given to 'individual_iterate' where each item in the iterator is a
# layer stored as a ModelLayer object
class ModelBuilder:

    MAX_LAYERS = 20

    def createiterator(self):
        return True

    #TODO

    # This function generates a conv layer, randomly setting parameters that don't affect other layers, eg. input size.
    # The setting of these parameters can be done layer during model generation when we have information on the other
    # layers
    def generateconvolutionallayer(self):
        #TODO, this would be generated randomly
        filters = 1
        kernel_size = [3,3]
        strides = 1

        # The input size is set to 0 as this will be set later during model generation when we have information on
        # the layers that preceed this convolutional layer
        return Conv2DLayer(filters, kernel_size, strides, 0)


### BLOCKS

# This class represents a block of a model, for example, a feature extraction block that would be some sort of combination
# of convolutional layer(s) and maybing pooling. Another example of a block would be a classification block which could be
# flattening followed by a dense layer
class ArchitecureBlock:

    def __init__(self):
        self.layer_count = 0

class FeatureExtractionBlock(ArchitecureBlock):

    some_stuff = 1

    def __init__(self):
        self.layers = 0;

    def add_random_feature_layers(self):
        #Some sort of randomness:
        # -> landing at this for example
        self.layers.append(ModelBuilder.generateconvolutionallayer())

class ClassificationBlock(ArchitecureBlock):

    some_stuff = 1

### MODEL OF BLOCKS

# We want to take a random, and valid (eg, don't start with classification blocks), combination of blocks and create an
# architecture

class BlockArchitecture:

    def __init__(self):
        self.blocks = []

    def generateblocksequence(self):
        # Do randomness and append a combination of blocks to self.blocks and then return a compiled list of all layers
        # TODO

        return itertools.chain(block.layers for block in self.blocks)