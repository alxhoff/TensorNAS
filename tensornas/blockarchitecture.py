from tensornas.block import Block
import numpy as np
import tensorflow as tf


class BlockArchitecture(Block):
    """
    A block architectures, eg. a classification architecture is one that provides a specified
    number of probability outputs that are used in the classification of some input.
    The abstract block architecture class defines the methods that must be implemented to allow for a type of block
    architecture to be created, namely what sort of sub-blocks the block architecture can generate.
    """

    def get_keras_model(self, optimizer, loss, metrics):
        layers = self.get_keras_layers()
        model = tf.keras.Sequential()
        for layer in layers:
            model.add(layer)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def evaluate(
        self,
        train_data,
        train_labels,
        test_data,
        test_labels,
        epochs,
        batch_size,
        optimizer,
        loss,
        metrics,
    ):
        model = self.get_keras_model(optimizer=optimizer, loss=loss, metrics=metrics)
        model.fit(x=train_data, y=train_labels, epochs=epochs, batch_size=batch_size)
        model.summary()
        ret = [
            int(
                np.sum(
                    [tf.keras.backend.count_params(p) for p in model.trainable_weights]
                )
            ),
            int(
                np.sum(
                    [
                        tf.keras.backend.count_params(p)
                        for p in model.non_trainable_weights
                    ]
                )
            ),
            model.evaluate(test_data, test_labels)[1] * 100,
        ]
