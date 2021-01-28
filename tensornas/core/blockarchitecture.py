import numpy as np
import tensorflow as tf

from tensornas.core.block import Block


class BlockArchitecture(Block):
    """
    A block architectures, eg. a classification architecture is one that provides a specified
    number of probability outputs that are used in the classification of some input.
    The abstract block architecture class defines the methods that must be implemented to allow for a type of block
    architecture to be created, namely what sort of sub-blocks the block architecture can generate.
    """

    def get_keras_model(self, optimizer, loss, metrics):
        inp = tf.keras.Input(shape=self.input_shape)
        out = self.get_keras_layers(inp)
        model = tf.keras.Model(inp, out)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def evaluate(
        self,
        train_data,
        train_labels,
        test_data,
        test_labels,
        epochs,
        steps,
        batch_size,
        optimizer,
        loss,
        metrics,
        filename=None,
    ):
        try:
            model = self.get_keras_model(
                optimizer=optimizer, loss=loss, metrics=metrics
            )
            model.summary()
            if filename:
                from tensornas.core.util import save_model

                save_model(model, filename)

            if batch_size == -1:
                model.fit(
                    x=train_data,
                    y=train_labels,
                    epochs=epochs,
                    steps_per_epoch=steps,
                    verbose=1,
                )
            else:
                model.fit(
                    x=train_data,
                    y=train_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    steps_per_epoch=steps,
                    verbose=1,
                )
        except Exception as e:
            import math

            print("Error fitting model, {}".format(e))
            return [math.inf, 0]
        params = int(
            np.sum([tf.keras.backend.count_params(p) for p in model.trainable_weights])
        ) + int(
            np.sum(
                [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
            )
        )

        accuracy = model.evaluate(test_data, test_labels)[1] * 100

        return params, accuracy
