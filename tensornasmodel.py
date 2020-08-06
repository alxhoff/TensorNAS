import tensorflow as tf
import keras
from tensornaslayers import *
import numpy as np


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

        # Here we pull our ModelLayer objects from the iterator containing our architecture
        for layer in layer_iterator:
            self._addlayer(layer.name, layer.args)

    def print(self):
        for layer in self.layers:
            layer.print()

    def loadlayersfromjson(self, json):
        if json:
            for x in range(len(json.keys())):
                layer = json.get(str(x))
                name = layer.get("name")
                args = layer.get("args")
                self.addlayer(name, args)

    def _addlayer(self, name, args):
        if name == "Conv2D":
            filters = args.get(Conv2DArgs.FILTERS.name, 1)
            kernel_size = tuple(args.get(Conv2DArgs.KERNEL_SIZE.name, (3, 3)))
            strides = tuple(args.get(Conv2DArgs.STRIDES.name, (1, 1)))
            input_size = tuple(args.get(Conv2DArgs.INPUT_SIZE.name))
            activation = args.get(Conv2DArgs.ACTIVATION.name, Activations.RELU.value)
            dilation_rate = tuple(args.get(Conv2DArgs.DILATION_RATE.name, (1, 1)))
            padding = args.get(Conv2DArgs.PADDING.name, PaddingArgs.VALID.value)
            self.layers.append(
                Conv2DLayer(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    input_size=input_size,
                    activation=activation,
                    dilation_rate=dilation_rate,
                    padding=padding,
                )
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
                self.layers.append(
                    MaxPool2DLayer(
                        pool_size=pool_size, strides=strides, padding=padding
                    )
                )
                if self.verbose:
                    print(
                        "Created {} layer with {} pool size and {} stride size".format(
                            name, pool_size, strides
                        )
                    )
            else:
                strides = tuple(args.get(MaxPool2DArgs.STRIDES.name, (1, 1, 1)))
                self.layers.append(
                    MaxPool3DLayer(
                        pool_size=pool_size, strides=strides, padding=padding
                    )
                )
                if self.verbose:
                    print(
                        "Created {} layer with {} pool size and {} stride size".format(
                            name, pool_size, strides
                        )
                    )
        elif name == "Reshape":
            target_shape = args.get(ReshapeArgs.TARGET_SHAPE.name)
            self.layers.append(ReshapeLayer(target_shape))
            if self.verbose:
                print(
                    " Created {} layer with {} target shape".format(name, target_shape)
                )
        elif name == "Dense":
            units = args.get(DenseArgs.UNITS.name)
            activation = getattr(
                tf.nn, args.get(DenseArgs.ACTIVATION.name, Activations.LINEAR.value)
            )
            self.layers.append(DenseLayer(units, activation))
            if self.verbose:
                print(
                    "Created {} layer with {} units and {} activation".format(
                        name, units, activation
                    )
                )
        elif name == "Flatten":
            self.layers.append(FlattenLayer())
            if self.verbose:
                print("Create {} layer".format(name))
        elif name == "Dropout":
            rate = args.get(DropoutArgs.RATE.name)
            self.layers.append(Dropout(rate))
            if self.verbose:
                print("Created {} layers with {} rate".format(name, rate))

    def _gettfmodel(self):
        model = keras.Sequential()
        for layer in self.layers:
            try:
                model.add(layer.getkeraslayer())
            except Exception as e:
                print("Hello")
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        if self.verbose:
            model.summary()
        return model

    def evaluate(
        self, train_data, train_labels, test_data, test_labels, epochs, batch_size
    ):
        model = self._gettfmodel()
        model.fit(x=train_data, y=train_labels, epochs=epochs, batch_size=batch_size)
        self.param_count = int(
            np.sum([keras.backend.count_params(p) for p in model.trainable_weights])
        ) + int(
            np.sum([keras.backend.count_params(p) for p in model.non_trainable_weights])
        )
        self.accuracy = model.evaluate(test_data, test_labels)[1] * 100

        return self.param_count, self.accuracy
