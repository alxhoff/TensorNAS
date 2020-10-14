import tensorflow as tf
import numpy as np


class Model:
    def __init__(
        self,
        block_architecture_class,
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        verbose=False,
    ):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.verbose = verbose
        self.accuracy = None
        self.param_count = None
        self.block_architecture = type(block_architecture_class)()

        self.print()

    def print(self):
        self.block_architecture.print()

    def loadlayersfromjson(self, json):
        if json:
            for x in range(len(json.keys())):
                layer = json.get(str(x))
                name = layer.get("name")
                args = layer.get("args")
                self.addlayer(name, args)

    def get_keras_model(self):
        model = tf.keras.Sequential()
        layers = self.block_architecture.get_keras_layers()
        for layer in layers:
            model.add(layer.get_keras_layer())
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        if self.verbose:
            model.summary()
        return model

    def validate(self):
        # TODO
        pass

    def evaluate(
        self, train_data, train_labels, test_data, test_labels, epochs, batch_size
    ):
        model = self.get_keras_model()
        model.fit(x=train_data, y=train_labels, epochs=epochs, batch_size=batch_size)
        self.param_count = int(
            np.sum([tf.keras.backend.count_params(p) for p in model.trainable_weights])
        ) + int(
            np.sum(
                [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
            )
        )
        self.accuracy = model.evaluate(test_data, test_labels)[1] * 100

        return self.param_count, self.accuracy
