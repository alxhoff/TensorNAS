from TensorNAS.Core.Block import Block


class BlockArchitecture(Block):
    """
    A block architectures, eg. a classification architecture is one that provides a specified
    number of probability outputs that are used in the classification of some input.
    The abstract block architecture class defines the methods that must be implemented to allow for a type of block
    architecture to be created, namely what sort of sub-blocks the block architecture can generate.
    """

    MAX_BATCH_SIZE = 128

    def __init__(self, input_shape, parent_block, layer_type, batch_size, optimizer):
        self.param_count = 0
        self.accuracy = 0

        from TensorNAS.Optimizers import GetOptimizer

        self.opt = GetOptimizer(optimizer_name=optimizer)

        self.batch_size = batch_size

        super().__init__(
            input_shape=input_shape, parent_block=parent_block, layer_type=layer_type
        )

    def _mutate_optimizer_hyperparameters(self, verbose):
        if self.opt:
            self.opt.mutate(verbose)

    def _mutate_batch_size(self, verbose):
        from TensorNAS.Core.Util import mutate_int_square

        self.batch_size = mutate_int_square(self.batch_size, 1, self.MAX_BATCH_SIZE)

    def get_keras_model(self, optimizer, loss, metrics):
        import tensorflow as tf

        inp = tf.keras.Input(shape=self.input_shape)
        out = self.get_keras_layers(inp)
        model = tf.keras.Model(inp, out)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=True)
        return model

    def evaluate(
        self,
        train_data,
        train_labels,
        test_data,
        test_labels,
        epochs,
        batch_size,
        loss,
        metrics,
        test_name=None,
        model_name=None,
        use_GPU=True,
        q_aware=False,
        logger=None,
    ):
        import numpy as np

        if use_GPU:
            from TensorNAS.Tools.TensorFlow import GPU as GPU

            GPU.config_GPU()
        else:
            import os

            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        opt = self.opt.get_optimizer()

        try:
            model = self.get_keras_model(
                optimizer=opt,
                loss=loss,
                metrics=metrics,
            )
        except Exception as e:
            print("Error getting keras model: {}".format(e))
            return np.inf, 0

        if q_aware:
            try:
                import tensorflow_model_optimization as tfmot

                q_model = tfmot.quantization.keras.quantize_model(model)

                q_model.compile(optimizer=opt, loss=loss, metrics=metrics)
                model = q_model
            except Exception as e:
                print("Error getting QA model: {}".format(e))

        try:
            if not batch_size > 0:
                model.fit(
                    x=train_data,
                    y=train_labels,
                    epochs=epochs,
                    verbose=1,
                )
            else:
                import tensorflow as tf

                early_stopper = tf.keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=1, mode="max"
                )
                model.fit(
                    x=train_data,
                    y=train_labels,
                    validation_data=(test_data, test_labels),
                    epochs=epochs,
                    batch_size=self.batch_size,
                    verbose=1,
                    callbacks=[early_stopper],
                )
        except Exception as e:
            print("Error fitting model, {}".format(e))
            return np.inf, 0

        try:
            if test_name and model_name:
                from TensorNAS.Core.Util import save_model, save_block_architecture

                save_model(model, test_name, model_name, logger)
                save_block_architecture(self, test_name, model_name, logger)
        except Exception as e:
            if logger:
                logger.log("Error running/saving model:{}, {}".format(model_name, e))

        from tensorflow.keras.backend import count_params

        params = int(np.sum([count_params(p) for p in model.trainable_weights])) + int(
            np.sum([count_params(p) for p in model.non_trainable_weights])
        )
        if params == 0:
            params = np.inf

        try:
            accuracy = model.evaluate(test_data, test_labels)[1] * 100
        except Exception as e:
            accuracy = 0
            print("Error evaluating model: {}".format(e))

        return params, accuracy
