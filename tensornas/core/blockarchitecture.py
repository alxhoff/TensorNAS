import numpy as np

from tensornas.core.block import Block


class BlockArchitecture(Block):
    """
    A block architectures, eg. a classification architecture is one that provides a specified
    number of probability outputs that are used in the classification of some input.
    The abstract block architecture class defines the methods that must be implemented to allow for a type of block
    architecture to be created, namely what sort of sub-blocks the block architecture can generate.
    """

    def __init__(self, input_shape, parent_block, layer_type):
        self.param_count = 0
        self.accuracy = 0

        super().__init__(
            input_shape=input_shape, parent_block=parent_block, layer_type=layer_type
        )

    def get_keras_model(self, optimizer, loss, metrics):
        import tensorflow as tf

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
        test_name=None,
        model_name=None,
        use_GPU=True,
    ):
        import tensorflow as tf

        if use_GPU:
            from tensornas.tools.tensorflow import GPU as GPU

            GPU.config_GPU()

        try:
            model = self.get_keras_model(
                optimizer=optimizer, loss=loss, metrics=metrics
            )
            if test_name and model_name:
                from tensornas.core.util import save_model

                save_model(model, test_name, model_name)

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

            print("Error fitting model, {}".format(e))
            return np.inf, 0
        params = int(
            np.sum([tf.keras.backend.count_params(p) for p in model.trainable_weights])
        ) + int(
            np.sum(
                [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
            )
        )
        if params == 0:
            params = np.inf
        accuracy = model.evaluate(test_data, test_labels)[1] * 100
        return params, accuracy
