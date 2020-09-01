import random
from itertools import *

import tensorflow as tf
import keras
import numpy as np

class ModelLayer:
    "Common layer properties"

    def __init__(self, name, args=None):
        print("Entered Individual_gen.py")
        self.name = name
        if args:
            self.args = args
        else:
            self.args = {}

    def getname(self):
        return self.name

    def getargs(self):
        return self.args

    def print(self):
        print("{} ->".format(self.name))
        for param_name, param_value in self.args.items():
            print("{}: {}".format(param_name, param_value))

class Conv2DLayer(ModelLayer):
    
    def __init__(self, filters, kernel_size, strides, input_size):
        super().__init__("Conv2D")
        print(input_size)
        self.args={}
        #x = tf.random.normal(input_size)
        self.args["Conv2DArgs.FILTERS.name"] = filters
        self.args["Conv2DArgs.KERNEL_SIZE.name"] = kernel_size
        self.args["Conv2DArgs.STRIDES.name"] = strides
        self.args["Conv2DArgs.INPUT_SIZE.name"] = input_size
        print("filter {}, kernel_size {}, strides {}".format(filters,kernel_size,strides))

    def mutate(self):
        pass

    def validate(self):
        if not 0 > self.args["Conv2DArgs.FILTERS.name"]:
            return False
        if not 0 > self.args["Conv2DArgs.KERNEL_SIZE.name"]:
            return False
        if not self.args["Conv2DArgs.KERNEL_SIZE.name"] < self.args["Conv2DArgs.INPUT_SIZE.name"]:
            return False
        


    #Conv2DLayer(12,[2,2],[1,1],784).getkeraslayer()


        # more checks here

    def getkeraslayer(self):
        return keras.layers.Conv2D(
            self.args["Conv2DArgs.FILTERS.name"],
            kernel_size=self.args["Conv2DArgs.KERNEL_SIZE.name"],
            strides=self.args["Conv2DArgs.STRIDES.name"],
            input_shape=self.args["Conv2DArgs.INPUT_SIZE.name"],
        )


class MaxPool2DLayer(ModelLayer):
    
    def __init__(self, pool_size, strides):
        self.args={}
        #super().__init__("MaxPool2D")
        self.args["MaxPool2DArgs.POOL_SIZE.name"] = pool_size
        self.args["MaxPool2DArgs.STRIDES.name"] = strides

    def getkeraslayer(self):
        return keras.layers.MaxPool2D(
            pool_size=self.args["MaxPool2DArgs.POOL_SIZE.name"],
            strides=self.args["MaxPool2DArgs.STRIDES.name"],
        )




class MaxPool3DLayer(MaxPool2DLayer):
    def __init__(self, pool_size, strides):
        self.args={}
        super(MaxPool2DLayer, self).__init__("MaxPool3D")
        super().__init__(pool_size, strides)

    def getkeraslayer(self):
        return keras.layers.MaxPool3D(
            pool_size=self.args["MaxPool2DArgs.POOL_SIZE.name"],
            strides=self.args["MaxPool2DArgs.STRIDES.name"],
        )
 

class ReshapeLayer(ModelLayer):
    def __init__(self, target_shape):
        self.args={}
        #super().__init__("Reshape")
        self.args["ReshapeArgs.TARGET_SHAPE.name"] = target_shape

    def getkeraslayer(self):
        target_shape = self.args["ReshapeArgs.TARGET_SHAPE.name"]
        return keras.layers.Reshape(target_shape)


class DenseLayer(ModelLayer):
    def __init__(self, units, activation):
        self.args={}
        super().__init__("Dense")
        self.args["DenseArgs.UNITS.name"] = units
        self.args["DenseArgs.ACTIVATION.name"] = activation

    def getkeraslayer(self):
        return keras.layers.Dense(
            self.args["DenseArgs.UNITS.name"],
            activation=self.args["DenseArgs.ACTIVATION.name"],
        )


class FlattenLayer(ModelLayer):
    def __init__(self):
        self.args={}
        super().__init__("Flatten")

    def getkeraslayer(self):
        return keras.layers.Flatten()


class Dropout(ModelLayer):
    def __init__(self, rate):
        self.args={}
        super().__init__("Dropout")
        self.args["DropoutArgs.RATE.name"] = rate

    def getkeraslayer(self):
        rate = self.args["DropoutArgs.RATE.name"]
        return keras.layers.Dropout(rate)

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

    def gen_int(self, min_bound, max_bound):
        return random.randrange(min_bound, max_bound + 1)

    def get_kernel_size(self,input_size):
        return random.randint(1,input_size)

    def get_strides(self):
        return random.randint(1,28)

    def gen_poolsize(self):
        size= random.randint(1,28)
        return [size,size]


    # This function generates a conv layer, randomly setting parameters that don't affect other layers, eg. input size.
    # The setting of these parameters can be done layer during model generation when we have information on the other
    # layers
    def generateconvolutionallayer(self,input_size):
        #TODO, this would be generated randomly
        print("entered generate conv layer")
        #input_size= input_size[0]*input_size[1]*input_size[2]
        print(input_size[0])
        filters = self.gen_int(1,input_size[0])
        kernel_size = self.get_kernel_size(input_size[0])
        strides = self.get_strides()
        if not (Conv2DLayer(filters, kernel_size, strides, input_size).validate):
            return False
        else:
            return Conv2DLayer(filters, kernel_size, strides, input_size).getkeraslayer()
        #print("filter {}, kernel_size {}, strides {}".format(filter,kernel_size,strides) )

        # The input size is set to 0 as this will be set later during model generation when we have information on
        # the layers that preceed this convolutional layer
        

    def generatepoolinglayer(self):
        print("entered generate maxpool layer")
        pool_size=self.gen_poolsize()
        strides=self.get_strides()
        return MaxPool2DLayer(pool_size,strides).getkeraslayer()

    def generatedenselayer(self):
        print("entered generate dense layer")
        nodes = random.randint(1,512)
        return DenseLayer(nodes,"relu").getkeraslayer()

    def generatedropoutlayer(self):
        print("entered generate dropout layer")
        dropout_rate=random.random()
        return Dropout(dropout_rate).getkeraslayer()

    #def generateflattenlayer(self,prev_output):
        #return prev_output[0]*prev_output[1]*prev_output[2]



### BLOCKS

# This class represents a block of a model, for example, a feature extraction block that would be some sort of combination
# of convolutional layer(s) and maybing pooling. Another example of a block would be a classification block which could be
# flattening followed by a dense layer
class ArchitecureBlock:

    def __init__(self):
        self.layer_count = 0

class FeatureExtractionBlock(ArchitecureBlock):

    #some_stuff = 1

    def __init__(self):
        self.layers = []
    def add_random_feature_layers(self,feature_layers,input_shape):
        #Some sort of randomness:
        # -> landing at this for example
        print("Feature Extraction block layers iterator: {}".format(feature_layers))
        m=ModelBuilder()
        #print(N)
        print()
        while feature_layers>=0:
            #print(m.generateconvolutionallayer(784))
            print("Calling conv layer constructor")
            #print(m.generateconvolutionallayer(784))
            if not (m.generateconvolutionallayer):
                return False
            else:
                self.layers.append(m.generateconvolutionallayer(input_shape))
            #print("printing layers in add_random_feature_conv {}".format(self.layers))
            
            if feature_layers%2==0:
                print("Calling pool layer constructor")
                #print(m.generatepoolinglayer())
                self.layers.append(m.generatepoolinglayer())
                #print("printing layers in add_random_feature_pool{}".format(self.layers))
            feature_layers-=1
        return self.layers
    #validations at features extraction block

class ClassificationBlock(ArchitecureBlock):

    some_stuff = 1
    
    def __init__(self):
        self.layers = []
    def add_random_classification_layers(self,classfication_layers):
        m=ModelBuilder()
        #Some sort of randomness:
        # -> landing at this for example
        self.layers.append(m.generatedenselayer())
        self.layers.append(m.generatedropoutlayer())
        #print("printing layers in add_random_classification_layers{}".format(self.layers))
        return self.layers

        #self.layers.append(ModelBuilder().generateflattenlayer())

        #validations at features extraction block

### MODEL OF BLOCKS

# We want to take a random, and valid (eg, don't start with classification blocks), combination of blocks and create an
# architecture

class BlockArchitecture:

    def __init__(self):
        self.blocks = []

    def generateblocksequence(self,feature_layers,classification_layers,feature_block,input_shape):
        count_individual=1
        self.blocks=[]
        print("Created Individual {}".format(count_individual))
        # Do randomness and append a combination of blocks to self.blocks and then return a compiled list of all layers
        # N * Conv Layer : 'N' indicates repetition of Conv Layer
        # M*( N*Conv --> Pool ?) : 'M' indicates repition of Feature extraction blocks
        # (FC--> Relu)*k --> indicates repitition of Classification blocks
        # N_max = 4
        M_min=1
        M_max=3
        K_min=1
        K_max = 3
        M=1
        print("No. of feature blocks: {}".format(feature_block))
        #K=random.randint(K_min, K_max)
        #print("No. of classification Blocks: {}".format(K))
        f=FeatureExtractionBlock()
        c=ClassificationBlock()
        while feature_block!=0:
            #print("printing appeneded blocks {}".format(f.add_random_feature_layers()))
            if not f.add_random_feature_layers(feature_layers,input_shape):
                return False
            else:
                self.blocks.append(f.add_random_feature_layers(feature_layers,input_shape))

            feature_block-=1
        #while K!=0:
        self.blocks.append(c.add_random_classification_layers(classification_layers))
            #K -= 1
        #validations at architecture level
        for block in self.blocks:
            #print(block)
            pass
        #print("Priya")
        #for block in self.blocks:
                #print("printing layers {} ".format(itertools.chain(block)))
        count_individual +=1
        print("returning from generate block sequence")
        #print(list(chain.from_iterable((self.blocks))))
        print(len(list(chain.from_iterable((self.blocks)))))
        return chain.from_iterable((self.blocks))
#b= BlockArchitecture()
#print(b.generateblocksequence())
#testing each funtion from layer to blocks to architecture
#model= ModelBuilder()
#print(model.generateconvolutionallayer(784)) 
#print(model.generatepoolinglayer())  
#print(model.generatedenselayer())
#print(model.generatedropoutlayer())
#feature_block= FeatureExtractionBlock()
#print("printing feature blocks {} ".format(feature_block.layers))