from tensornas.block import Block
import tensorflow as tf
from tensornas.layerargsbuilder import *


class LayerBlock(Block):
    """
    Layer blocks represent the lowest level blocks of the block architecture, they exist to house a single
    keras layer with which they interface to provide the appropriate information to the higher level
    blocks, eg. input/output shapes.

    To manage the keras layer the block contains a ModelLayer object which works to manage the properties of the
    tensorflow/keras layer which can then be generated when required.
    """

    MAX_SUB_BLOCKS = 0
    SUB_BLOCK_TYPES = None

    def __init__(self, input_shape, parent_block, layer_type):
        self.args = None
        self.layer_type = layer_type
        self.create_layer(layer_type)

        super().__init__(input_shape=input_shape, parent_block=parent_block)

    def mutate(self):
        self.layer.mutate()

    def generate_constrained_output_sub_blocks(self, input_shape):
        pass

    def generate_constrained_input_sub_blocks(self, input_shape):
        pass

    def generate_random_sub_block(self, layer_type):
        pass

    def output_shape(self):
        self.layer.output_shape()

    def validate(self):
        self.layer.validate()

    def get_keras_layers(self):
        new_layer = None

        if self.layer_type == "Conv2D":
            filters = self.args.get(Conv2DArgs.FILTERS.name, 1)
            kernel_size = tuple(self.args.get(Conv2DArgs.KERNEL_SIZE.name, (3, 3)))
            strides = tuple(self.args.get(Conv2DArgs.STRIDES.name, (1, 1)))
            input_size = tuple(
                self.args.get(Conv2DArgs.INPUT_SHAPE.name, self.input_shape)
            )
            activation = self.args.get(
                Conv2DArgs.ACTIVATION.name, Activations.RELU.value
            )
            dilation_rate = tuple(self.args.get(Conv2DArgs.DILATION_RATE.name, (1, 1)))
            padding = self.args.get(Conv2DArgs.PADDING.name, PaddingArgs.VALID.value)

            new_layer = tf.Keras.Conv2DLayer(
                input_shape=input_size,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                activation=activation,
                dilation_rate=dilation_rate,
                padding=padding,
            )
            if self.verbose:
                print(
                    "Created {} layer with {} filters, {} kernel size, {} stride size and {} input size".format(
                        self.layer_type, filters, kernel_size, strides, input_size
                    )
                )

        elif self.layer_type == "MaxPool2D":

            pool_size = tuple(
                self.args.get(MaxPool2DArgs.POOL_SIZE.self.layer_type, (1, 1))
            )
            padding = self.args.get(
                MaxPool2DArgs.PADDING.self.layer_type, PaddingArgs.VALID.value
            )
            strides = tuple(
                self.args.get(MaxPool2DArgs.STRIDES.self.layer_type, (1, 1))
            )
            new_layer = tf.Keras.MaxPool2DLayer(
                input_shape=self.input_shape,
                pool_size=pool_size,
                strides=strides,
                padding=padding,
            )
            if self.verbose:
                print(
                    "Created {} layer with {} pool size and {} stride size".format(
                        self.layer_type, pool_size, strides
                    )
                )
        elif self.layer_type == "MaxPool3D":

            pool_size = tuple(
                self.args.get(MaxPool2DArgs.POOL_SIZE.self.layer_type, (1, 1))
            )
            padding = self.args.get(
                MaxPool2DArgs.PADDING.self.layer_type, PaddingArgs.VALID.value
            )
            strides = tuple(
                self.args.get(MaxPool2DArgs.STRIDES.self.layer_type, (1, 1, 1))
            )
            new_layer = tf.Keras.MaxPool3DLayer(
                input_shape=self.input_shape,
                pool_size=pool_size,
                strides=strides,
                padding=padding,
            )
            if self.verbose:
                print(
                    "Created {} layer with {} pool size and {} stride size".format(
                        self.layer_type, pool_size, strides
                    )
                )
        elif self.layer_type == "Reshape":
            target_shape = tuple(
                self.args.get(
                    ReshapeArgs.TARGET_SHAPE.self.layer_type, self.input_shape
                )
            )
            new_layer = tf.Keras.ReshapeLayer(
                input_shape=self.input_shape, target_shape=target_shape
            )
            if self.verbose:
                print(
                    " Created {} layer with {} target shape".format(
                        self.layer_type, target_shape
                    )
                )
        elif self.layer_type == "Dense":
            units = self.args.get(DenseArgs.UNITS.self.layer_type)
            activation = getattr(
                tf.nn,
                self.args.get(
                    DenseArgs.ACTIVATION.self.layer_type, Activations.LINEAR.value
                ),
            )

            new_layer = tf.Keras.DenseLayer(
                input_shape=self.input_shape, units=units, activation=activation
            )
            if self.verbose:
                print(
                    "Created {} layer with {} units and {} activation".format(
                        self.layer_type, units, activation
                    )
                )
        elif self.layer_type == "Flatten":
            new_layer = tf.Keras.FlattenLayer(input_shape=self.input_shape)
            if self.verbose:
                print("Create {} layer".format(self.layer_type))
        elif self.layer_type == "Dropout":
            rate = self.args.get(DropoutArgs.RATE.self.layer_type)
            new_layer = tf.Keras.DropoutLayer(input_shape=self.input_shape, rate=rate)
            if self.verbose:
                print("Created {} layers with {} rate".format(self.layer_type, rate))

        self.layers.append(new_layer)
        return new_layer.outputshape.get()

    def print_self(self):
        print("Layers: {}".format(self.layer_type))

    def create_layer(self, name, input_shape):
        new_layer = None

        if name == "Conv2D":
            self.args = gen_conv2d_args()

        elif name == "MaxPool2D":
            self.args = gen_pool2d_args()

        elif name == "MaxPool3D":
            self.args = gen_pool3d_args()

        elif name == "Reshape":
            self.args = gen_reshape_args()

        elif name == "Dense":
            self.args = gen_dense_args()

        elif name == "Dropout":
            self.args = gen_drop_args()
