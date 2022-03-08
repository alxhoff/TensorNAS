import gc
import math

from TensorNAS.Core.Block import Block

from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import Callback


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()


class BlockArchitecture(Block):
    """
    A block architectures, eg. a classification architecture is one that provides a specified
    number of probability outputs that are used in the classification of some input.
    The abstract block architecture class defines the methods that must be implemented to allow for a type of block
    architecture to be created, namely what sort of sub-blocks the block architecture can generate.
    """

    MAX_BATCH_SIZE = 128

    def __init__(self, input_shape, batch_size, optimizer):
        self.param_count = 0
        self.accuracy = 0
        self.batch_size = batch_size

        from TensorNAS.Optimizers import GetOptimizer

        self.optimizer = optimizer
        self.opt = GetOptimizer(optimizer_name=optimizer)

        super().__init__(input_shape=input_shape, parent_block=None)

    def _mutate_optimizer_hyperparameters(self, verbose):
        if self.opt:
            self.opt.mutate(verbose)

    def _mutate_batch_size(self, verbose):
        from TensorNAS.Core.Mutate import mutate_int_square

        self.batch_size = mutate_int_square(self.batch_size, 1, self.MAX_BATCH_SIZE)

    def get_keras_model(self, loss, metrics):
        import tensorflow as tf

        inp = tf.keras.Input(shape=self.input_shape)
        try:
            out = self.get_keras_layers(inp)
        except Exception as e:
            raise e

        if out != None:
            try:
                model = tf.keras.Model(inp, out)
                model.compile(
                    optimizer=self.opt.get_optimizer(),
                    loss=eval(loss),
                    metrics=metrics,
                )
                return model
            except Exception as e:
                raise e
        else:
            raise Exception("Getting keras model failed")

    def prepare_model(
        self,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        q_aware=False,
    ):

        try:
            model = self.get_keras_model(
                loss=loss,
                metrics=metrics,
            )
        except Exception as e:
            import traceback

            print("Error getting keras model: {}".format(e))
            traceback.format_exc()
            return None

        if q_aware:
            try:
                import tensorflow_model_optimization as tfmot

                q_model = tfmot.quantization.keras.quantize_model(model)
                q_model.compile(
                    optimizer=self.opt.get_optimizer(), loss=loss, metrics=metrics
                )
                model = q_model
            except Exception as e:
                print("Error getting QA model: {}".format(e))
                return None

        return model

    def save_model(self, model, test_name, model_name, logger):
        import numpy as np

        try:
            if test_name and model_name:
                from TensorNAS.Tools import save_model
                from TensorNAS.Tools import save_block_architecture

                save_model(model, test_name, model_name, logger)
                save_block_architecture(self, test_name, model_name, logger)
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            if logger:
                logger.log("Error running/saving model:{}, {}".format(model_name, e))

        from tensorflow.keras.backend import count_params

        params = int(np.sum([count_params(p) for p in model.trainable_weights])) + int(
            np.sum([count_params(p) for p in model.non_trainable_weights])
        )
        if params == 0:
            params = np.inf

        return params


class AreaUnderCurveBlockArchitecture(BlockArchitecture):
    def evaluate(
        self,
        train_data=None,
        train_labels=None,
        test_data=None,
        test_labels=None,
        train_generator=None,
        validation_generator=None,
        test_generator=None,
        validation_split=0.1,
        epochs=1,
        batch_size=1,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        test_name=None,
        model_name=None,
        q_aware=False,
        logger=None,
        steps_per_epoch=None,
        test_steps=None,
    ):
        from Demos import get_global

        if get_global("verbose"):
            verbose = 1
        else:
            verbose = 0

        model = self.prepare_model(loss=loss, metrics=metrics, q_aware=q_aware)

        model, params = self.train_model(
            model=model,
            train_data=train_data,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            test_name=test_name,
            model_name=model_name,
            logger=logger,
            verbose=verbose,
        )

        try:
            import numpy as np
            from tqdm import tqdm

            predictions = []
            for td in tqdm(test_data, total=len(test_data)):
                pred = model.predict(td)
                errors = np.mean(np.mean(np.square(td - pred), axis=1))
                predictions.append(errors)

            import sklearn.metrics as metrics

            auc = metrics.roc_auc_score(test_labels, predictions)
            return params, auc
        except Exception as e:
            auc = 0
            print("Error evaluating model: {}".format(e))

        return auc

    def train_model(
        self,
        model,
        train_data=None,
        validation_split=0.1,
        epochs=1,
        batch_size=1,
        test_name=None,
        model_name=None,
        logger=None,
        verbose=0,
    ):

        model.fit(
            x=train_data,
            y=train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=True,
            verbose=verbose,
        )

        params = self.save_model(
            model=model, test_name=test_name, model_name=model_name, logger=logger
        )

        return model, params


class ClassificationBlockArchitecture(BlockArchitecture):
    def __init__(self, input_shape, batch_size, optimizer, class_count):
        self.class_count = class_count

        super().__init__(
            input_shape=input_shape,
            batch_size=batch_size,
            optimizer=optimizer,
        )

    def evaluate(
        self,
        train_data=None,
        train_labels=None,
        test_data=None,
        test_labels=None,
        train_generator=None,
        validation_generator=None,
        test_generator=None,
        validation_steps=1,
        epochs=1,
        batch_size=1,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        test_name=None,
        model_name=None,
        q_aware=False,
        logger=None,
        steps_per_epoch=None,
        test_steps=None,
    ):
        from Demos import get_global

        if get_global("verbose"):
            verbose = 1
        else:
            verbose = 0

        model = self.prepare_model(loss=loss, metrics=metrics, q_aware=q_aware)

        if model == None:
            params = math.inf
            accuracy = 0
            return params, accuracy

        model, params = self.train_model(
            model=model,
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            train_generator=train_generator,
            validation_generator=validation_generator,
            validation_steps=validation_steps,
            epochs=epochs,
            batch_size=batch_size,
            test_name=test_name,
            model_name=model_name,
            logger=logger,
            steps_per_epoch=steps_per_epoch,
            verbose=verbose,
        )

        try:
            if (
                (train_data is not None)
                and (train_labels is not None)
                and (test_data is not None)
                and (test_labels is not None)
            ):
                accuracy = (
                    model.evaluate(
                        x=test_data,
                        y=test_labels,
                        batch_size=batch_size,
                        verbose=verbose,
                    )[1]
                    * 100
                )
            else:
                if not test_steps:
                    test_steps = len(test_generator) // batch_size
                accuracy = (
                    model.evaluate(
                        x=train_generator, batch_size=batch_size, steps=test_steps
                    )[1]
                    * 100
                )
        except Exception as e:
            accuracy = 0
            print("Error evaluating model: {}".format(e))

        gc.collect()

        return params, accuracy

    def train_model(
        self,
        model,
        train_data=None,
        train_labels=None,
        test_data=None,
        test_labels=None,
        train_generator=None,
        validation_generator=None,
        validation_split=0.1,
        validation_steps=1,
        epochs=1,
        batch_size=1,
        test_name=None,
        model_name=None,
        logger=None,
        steps_per_epoch=None,
        verbose=0,
    ):
        import numpy as np
        import tensorflow as tf

        early_stopper = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=1, mode="max"
        )

        try:
            if not batch_size > 0:
                if (train_data is not None) and (train_labels is not None):
                    model.fit(
                        x=train_data,
                        y=train_labels,
                        validation_split=validation_split,
                        epochs=epochs,
                        verbose=verbose,
                        callbacks=[early_stopper, ClearMemory()],
                    )
                else:
                    if not steps_per_epoch:
                        steps_per_epoch = len(train_generator) // batch_size
                    model.fit(
                        x=train_generator,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        callbacks=[early_stopper, ClearMemory()],
                    )
            else:

                if (
                    (train_data is not None)
                    and (train_labels is not None)
                    and (test_data is not None)
                    and (test_labels is not None)
                ):
                    model.fit(
                        x=train_data,
                        y=train_labels,
                        validation_data=(test_data, test_labels),
                        validation_steps=validation_steps,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=verbose,
                        callbacks=[early_stopper, ClearMemory()],
                    )
                else:
                    if not steps_per_epoch:
                        steps_per_epoch = len(train_generator) // batch_size
                    model.fit(
                        x=train_generator,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=verbose,
                        callbacks=[early_stopper, ClearMemory()],
                    )
        except Exception as e:
            print("Error fitting model, {}".format(e))
            return model, np.inf

        params = self.save_model(
            model=model, test_name=test_name, model_name=model_name, logger=logger
        )

        gc.collect()
        return model, params
