import random
from itertools import *

import tensorflow as tf
import keras
import numpy as np


import random
import itertools
import keras
from enum import Enum, auto
import numpy as np
from tensornasmutator import *
import tensornaslayers





# Model architecutre that contains a collection of blocks

# This class represents a block of a model, for example, a feature extraction block that would be some sort of combination
# of convolutional layer(s) and maybing pooling. Another example of a block would be a classification block which could be
# flattening followed by a dense layer
class BlockTypes(Enum):
    CLASSIFICATION_BLOCK = auto()
    FEATURE_EXTRACTION_BLOCK = auto()

class BlockArchitecture:
    MAX_BLOCKS = 2
    last_layer = False

    def __init__(self, input_shape,batch_size, no_classes):
        self.blocks = []
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.no_classes = no_classes

    def ind_gen(self):
        #for i in range(BlockArchitecture.MAX_BLOCKS):
        self.blocks.append(FeatureExtractionBlock(self.input_shape, self.batch_size,self.no_classes).gen_feature_extraction_block())
        #ClassificationBlock(self.input_shape, self.batch_size,self.no_classes).gen_classification_block()
        self.blocks.append(ClassificationBlock(self.input_shape, self.batch_size,self.no_classes).gen_classification_block())
        BlockArchitecture.last_layer = True
        self.blocks.append(ClassificationBlock(self.input_shape, self.batch_size,self.no_classes).gen_classification_block())
        BlockArchitecture.last_layer = False
        return self.blocks

    def get_iter(self):
        for block in self.blocks:
            print("block layers {} ".format(block.layers))
        return itertools.chain(block.layers for block in self.blocks)





### BLOCKS that contains collections of layers

class ArchitectureBlock:

    def __init__(self):
        self.layer_count = 0
        self.layers=[]

class FeatureExtractionBlockLayerTypes(Enum):
    CONV2D = auto()
    MAXPOOL2D = auto()

class FeatureExtractionBlock(ArchitectureBlock):
    MAX_FEATURE_LAYER_COUNT = 5
    layers=[]
    def __init__(self,input_shape,batch_size, no_classes):
        self.layers = []
        self.input_shape= input_shape
        self.block_size = batch_size
        self.no_classes = no_classes

    def gen_feature_extraction_block(self):
        self.layers.append(ModelBuilder().generateconvolutionallayer(self.input_shape))
        self.layers.append(ModelBuilder().generatepoolinglayer(self.input_shape))
        return self.layers





class ClassificationBlockLayerTypes(Enum):
    #FLATTEN = auto()
    DENSE = auto()
    DROPOUT = auto()


class ClassificationBlock(ArchitectureBlock):
    MAX_CLASSIFICATION_LAYER_COUNT = 2

    def __init__(self, input_shape, batch_size, no_classes):
        self.layers = []
        self.input_shape = input_shape
        self.batch_size  = batch_size
        self.no_classes = no_classes

    def gen_classification_block(self):
        if BlockArchitecture.last_layer:
            self.layers.append(ModelBuilder().generateoutputdenselayer(self.input_shape, self.no_classes))
        else:
            self.layers.append(ModelBuilder().generateflattenlayer(self.input_shape))
            self.layers.append(ModelBuilder().generatedenselayer(self.input_shape))
            self.layers.append(ModelBuilder().generatedropoutlayer(self.input_shape))
        # while len(self.layers) < ClassificationBlock.MAX_CLASSIFICATION_LAYER_COUNT:
        # layer_type = mutate_enum_i(ClassificationBlockLayerTypes)
        # if layer_type == 1:
        # self.layers.append(ModelBuilder().generatedenselayer(self.input_shape))
        # elif layer_type == 2:
        # self.layers.append(ModelBuilder().generatedropoutlayer(self.input_shape))
        # self.layers.append(ModelBuilder().generatereshapelayer(self.input_shape, [self.batch_size, -1]))

        return self.layers

    def validate_block(self):
        for layer, next_layers in zip(self.layers, self.layers[1:]):
            if not layer.calcoutputshape == next_layers.input_shape:
                next_layers.input_shape = layer.calcoutputshape

            if layer.name == "Dense":
                while not layer._units <= layer.input_shape:
                    layer._mutate_units

            if layer.name == "Reshape":
                pass
            if layer.name == "Flatten":
                pass
            if layer.name == "Dropout":
                pass
        return True







class ModelBuilder:
    MAX_LAYERS = 20

    def createiterator(self):
        return True

    def gen_int(self, min_bound, max_bound):
        return random.randrange(min_bound, max_bound + 1)

    def get_kernel_size(self, input_size):
        kernel_size = random.randint(1, input_size)
        return [kernel_size, kernel_size]

    def get_strides(self, max_bound):
        stride_size= random.randint(1, max_bound)
        return [stride_size, stride_size]

    def gen_poolsize(self, max_bound):
        size = random.randint(1, max_bound)
        return [size, size]

    # This function generates a conv layer, randomly setting parameters that don't affect other layers, eg. input size.
    # The setting of these parameters can be done layer during model generation when we have information on the other
    # layers
    def generateconvolutionallayer(self, input_shape):
        input_shape = input_shape
        filters = self.gen_int(1, input_shape[0]/2)
        kernel_size = self.get_kernel_size(input_shape[0]/2)
        strides = [1,1]
        padding = "valid"
        # we only get repaired layer out of validate and return the layer object to be built into .getkeraslayer()
        return  tensornaslayers.Conv2DLayer(input_shape,filters, kernel_size, strides,padding )


        # The input size is set to 0 as this will be set later during model generation when we have information on
        # the layers that preceed this convolutional layer

    def generatepoolinglayer(self,input_shape):
        pool_size = self.gen_poolsize(input_shape[0]/2)
        strides = self.get_strides(input_shape[0]/2)
        input_shape= input_shape
        #if tensornaslayers.MaxPool2DLayer(input_shape, pool_size).validate:
        return tensornaslayers.MaxPool2DLayer(input_shape, pool_size)

    def generateflattenlayer(self, input_shape):
        input_shape = input_shape
        #if tensornaslayers.FlattenLayer(input_shape).validate:
        return tensornaslayers.FlattenLayer(input_shape)

    def generatedenselayer(self, input_shape):
        nodes = random.randint(1, 512)
        input_shape= input_shape
        #if tensornaslayers.DenseLayer(input_shape, nodes, "relu").validate:
        return tensornaslayers.DenseLayer(input_shape,nodes, "relu")

    def generatedropoutlayer(self, input_shape):
        dropout_rate = random.random()
        input_shape = input_shape
        #if tensornaslayers.DropoutLayer(input_shape, dropout_rate):
        return tensornaslayers.DropoutLayer(input_shape,dropout_rate)

    def generateoutputdenselayer(self,input_shape,no_classes):
        input_shape = input_shape
        units = no_classes
        activation = "softmax"
        #if tensornaslayers.OutputDenseLayer(input_shape, units, activation):
        return tensornaslayers.OutputDenseLayer(input_shape, units, activation )

    def generatereshapelayer(self, input_shape, target_shape):
        input_shape = input_shape
        target_shape = target_shape
        return tensornaslayers.ReshapeLayer(input_shape, target_shape)

