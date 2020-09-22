import tensorflow as tf
import keras
from tensornaslayers import *
import numpy as np
import nasconstraints as nas

class TensorNASModel:
    def __init__(
        self,
        layer_iterator,
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        verbose=False,
    ):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.layers = []
        self.verbose = verbose
        self.accuracy = None
        self.param_count = None
        cur_input_shape = None



        # Here we pull our ModelLayer objects from the iterator containing our architecture
        for layer in layer_iterator:
            for every_layer in layer:
                if not cur_input_shape:
                    cur_input_shape = every_layer.args.get("INPUT_SHAPE")
                    if every_layer.validate and (not every_layer.calcoutputshape()[0] < 0):
                        cur_input_shape = self._addlayer(
                            name=every_layer.name, args=every_layer.args, input_shape=cur_input_shape)
                else:
                    every_layer.inputshape.set(cur_input_shape)
                    if every_layer.validate():
                        cur_input_shape = self._addlayer(
                        name=every_layer.name, args=every_layer.args, input_shape=cur_input_shape
                )

        self.print()

    def print(self):
        for x, layer in enumerate(self.layers):
            print("-------------------------------------------")
            print("Layer #{}".format(x))
            layer.print()
            #print(layer)

    def loadlayersfromjson(self, json):

        if json:
            for x in range(len(json.keys())):
                layer = json.get(str(x))
                name = layer.get("name")
                args = layer.get("args")
                self.addlayer(name, args)

    # Input shape is given to add layer such that adding layers can check if the new layer can handle the output
    # shape of the previous layer. The shape required is tracked as the TensorNASModel builds the layers
    def _addlayer(self, name, args, input_shape=None):
        new_layer = None
        if name == "Conv2D":
            filters = args.get(Conv2DArgs.FILTERS.name, 1)
            kernel_size = tuple(args.get(Conv2DArgs.KERNEL_SIZE.name, (3, 3)))
            strides = tuple(args.get(Conv2DArgs.STRIDES.name, (1, 1)))
            args['INPUT_SHAPE']= input_shape
            input_size = tuple(args.get(Conv2DArgs.INPUT_SHAPE.name, input_shape))
            activation = args.get(Conv2DArgs.ACTIVATION.name, Activations.RELU.value)
            dilation_rate = tuple(args.get(Conv2DArgs.DILATION_RATE.name, (1, 1)))
            padding = args.get(Conv2DArgs.PADDING.name, PaddingArgs.VALID.value)

            new_layer = Conv2DLayer(
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
                        name, filters, kernel_size, strides, input_size
                    )
                )

        elif name == "MaxPool2D" or name == "MaxPool3D":
            pool_size = tuple(args.get(MaxPool2DArgs.POOL_SIZE.name, (1, 1)))
            padding = args.get(MaxPool2DArgs.PADDING.name, PaddingArgs.VALID.value)
            if name == "MaxPool2D":
                strides = tuple(args.get(MaxPool2DArgs.STRIDES.name, (1, 1)))
                new_layer = MaxPool2DLayer(
                    input_shape=input_shape,
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding,
                )
                if self.verbose:
                    print(
                        "Created {} layer with {} pool size and {} stride size".format(
                            name, pool_size, strides
                        )
                    )
            else:
                strides = tuple(args.get(MaxPool2DArgs.STRIDES.name, (1, 1, 1)))
                new_layer = MaxPool3DLayer(
                    input_shape=input_shape,
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding,
                )
                if self.verbose:
                    print(
                        "Created {} layer with {} pool size and {} stride size".format(
                            name, pool_size, strides
                        )
                    )
        elif name == "Reshape":
            target_shape = tuple(args.get(ReshapeArgs.TARGET_SHAPE.name, input_shape))
            new_layer = ReshapeLayer(input_shape=input_shape, target_shape=target_shape)
            if self.verbose:
                print(
                    " Created {} layer with {} target shape".format(name, target_shape)
                )
        elif name == "Dense":
            units = args.get(DenseArgs.UNITS.name)
            activation = getattr(
                tf.nn, args.get(DenseArgs.ACTIVATION.name, Activations.LINEAR.value)
            )

            new_layer = DenseLayer(
                input_shape=input_shape, units=units, activation=activation
            )
            if self.verbose:
                print(
                    "Created {} layer with {} units and {} activation".format(
                        name, units, activation
                    )
                )
        elif name == "Flatten":
            new_layer = FlattenLayer(input_shape=input_shape)
            if self.verbose:
                print("Create {} layer".format(name))
        elif name == "Dropout":
            rate = args.get(DropoutArgs.RATE.name)
            new_layer = DropoutLayer(input_shape=input_shape, rate=rate)
            if self.verbose:
                print("Created {} layers with {} rate".format(name, rate))
        elif name == "Output_Dense":
            units = args.get(DenseArgs.UNITS.name)
            activation = getattr(
                tf.nn, args.get(DenseArgs.ACTIVATION.name, Activations.SOFTMAX.value)
            )

            new_layer = DenseLayer(
                input_shape=input_shape, units=units, activation=activation
            )
            if self.verbose:
                print(
                    "Created Output {} layer with {} units and {} activation".format(
                        name, units, activation
                    )
                )

        self.layers.append(new_layer)
        return new_layer.outputshape.get()

    def _gettfmodel(self):
        model = keras.Sequential()
        for layer in self.layers:
            #print(layer)
            #for every_
            model.add(layer.getkeraslayer())
        #print(model.summary())
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        if self.verbose:
            model.summary()
        return model

    def evaluate(
            self, train_data, train_labels, test_data, test_labels, epochs, batch_size
    ):
        model = self._gettfmodel()
        #model._maybe_build((28,28,1))
        #print(model.summary())
        print(train_labels.shape)
        print(train_data.shape)
        model.fit(x=train_data, y=train_labels, epochs=epochs, batch_size=batch_size)
        print("Completed fit")

        self.param_count = int(
            np.sum([keras.backend.count_params(p) for p in model.trainable_weights])
        ) + int(
            np.sum([keras.backend.count_params(p) for p in model.non_trainable_weights])
        )
        self.accuracy = model.evaluate(test_data, test_labels)[1] * 100
        print("completed evaluate")

        return self.param_count, self.accuracy