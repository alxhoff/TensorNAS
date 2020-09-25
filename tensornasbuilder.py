import random
import tensornaslayers


class ModelBuilder:
    @staticmethod
    def get_kernel_size(input_size):
        kernel_size = random.randint(1, input_size)
        return [kernel_size, kernel_size]

    @staticmethod
    def get_strides(max_bound):
        stride_size = random.randint(1, max_bound)
        return [stride_size, stride_size]

    @staticmethod
    def gen_poolsize(max_bound):
        size = random.randint(1, max_bound)
        return [size, size]

    # This function generates a conv layer, randomly setting parameters that don't affect other layers, eg. input size.
    # The setting of these parameters can be done layer during model generation when we have information on the other
    # layers
    @staticmethod
    def generateconvolutionallayer(input_shape=(0, 0, 0)):
        input_shape = input_shape
        filters = ModelBuilder.gen_int(1, input_shape[0] / 2)
        kernel_size = ModelBuilder.get_kernel_size(input_shape[0] / 2)
        strides = [1, 1]
        padding = "valid"
        # we only get repaired layer out of validate and return the layer object to be built into .getkeraslayer()
        return tensornaslayers.Conv2DLayer(
            input_shape, filters, kernel_size, strides, padding
        )

        # The input size is set to 0 as this will be set later during model generation when we have information on
        # the layers that preceed this convolutional layer

    @staticmethod
    def generatepoolinglayer(input_shape=(0, 0, 0)):
        pool_size = ModelBuilder.gen_poolsize(input_shape[0] / 2)
        strides = ModelBuilder.get_strides(input_shape[0] / 2)
        input_shape = input_shape
        # if tensornaslayers.MaxPool2DLayer(input_shape, pool_size).validate:
        return tensornaslayers.MaxPool2DLayer(input_shape, pool_size)

    @staticmethod
    def generateflattenlayer(input_shape=(0, 0, 0)):
        input_shape = input_shape
        # if tensornaslayers.FlattenLayer(input_shape).validate:
        return tensornaslayers.FlattenLayer(input_shape)

    @staticmethod
    def generatedenselayer(input_shape=(0, 0, 0)):
        nodes = random.randint(1, 512)
        input_shape = input_shape
        # if tensornaslayers.DenseLayer(input_shape, nodes, "relu").validate:
        return tensornaslayers.DenseLayer(input_shape, nodes, "relu")

    @staticmethod
    def generatedropoutlayer(input_shape=(0, 0, 0)):
        dropout_rate = random.random()
        input_shape = input_shape
        # if tensornaslayers.DropoutLayer(input_shape, dropout_rate):
        return tensornaslayers.DropoutLayer(input_shape, dropout_rate)

    @staticmethod
    def generateoutputdenselayer(no_classes, input_shape=(0, 0, 0)):
        input_shape = input_shape
        units = no_classes
        activation = "softmax"
        # if tensornaslayers.OutputDenseLayer(input_shape, units, activation):
        return tensornaslayers.OutputDenseLayer(input_shape, units, activation)

    @staticmethod
    def generatereshapelayer(target_shape, input_shape=(0, 0, 0)):
        input_shape = input_shape
        target_shape = target_shape
        return tensornaslayers.ReshapeLayer(input_shape, target_shape)
