import gc
import math

import keras.models

import TensorNAS.Core.Individual
from TensorNAS.Core.Block import Block
from TensorNAS.Core.LayerMutations import layer_mutation
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import Callback


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()


from Demos import get_global

from enum import Enum, auto


class OptimizationGoal(Enum):

    ACCURACY_UP = auto()
    PARAMETERS_DOWN = auto()


class Mutation:
    def __init__(
        self,
        mutation_table_references,
        accuracy_diff=None,
        param_diff=None,
        mutation_function=None,
        mutation_note=None,
    ):

        # Initially mutations are added to block architectures with a list of references to the locations in child
        # blocks' mutation tables where the accuracy and param count differences must be added, these values cannot
        # be added until the model has been retrained after the mutation, the pending flag shows that a block
        # architecture has been mutated and needs to have the accuracy and param count diffs added into it's child
        # blocks' mutation tables.
        self.pending = True
        self.mutation_table_references = mutation_table_references
        self.mutation_function = mutation_function
        self.mutation_note = mutation_note
        self.accuracy_diff = accuracy_diff
        self.param_diff = param_diff

    def _update_q(self, delta, q_old):

        alpha = get_global("alpha")

        return alpha * delta + (1 - alpha) * q_old

    def propogate_mutation_results(self):

        for ref in self.mutation_table_references:
            # Updating Q values is done by first normalizing the values using the normalization values provided in
            # the config file, then updating the existing Q values using the formula
            # Q_n = alpha * delta + (1 - alpha) Q_n-1

            # Normalize
            # Assumes a single normalization vector and not a varying one
            normalization_vector = get_global("filter_function_args")[1][0]
            n_param_count = -self.param_diff / float(normalization_vector[0])
            n_acc = self.accuracy_diff / float(normalization_vector[1])

            # Update
            ref[0] = self._update_q(n_param_count, ref[0])
            ref[1] = self._update_q(n_acc, ref[1])

        self.pending = False


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
        self.prev_param_count = 0
        self.accuracy = 0
        self.prev_accuracy = 0
        self.mutations = []
        self.batch_size = batch_size
        self.optimization_goal = None

        from TensorNAS.Optimizers import GetOptimizer

        self.optimizer = optimizer
        self.opt = GetOptimizer(optimizer_name=optimizer)

        super().__init__(input_shape=input_shape, parent_block=None)

    def mutate(self, mutate_equally=True, mutation_probability=0.0, verbose=False):

        goal_index = 0
        if self.optimization_goal == OptimizationGoal.ACCURACY_UP:
            goal_index = 1

        return super().mutate(
            mutation_goal_index=goal_index,
            mutate_equally=mutate_equally,
            mutation_probability=mutation_probability,
            verbose=verbose,
        )

    @layer_mutation
    def _mutate_optimizer_hyperparameters(self, verbose=False):
        if self.opt:
            return self.opt.mutate(verbose)
        return "_mutate_optimizer_hyperparameters", "Null mutation"

    @layer_mutation
    def _mutate_batch_size(self, verbose=False):
        from TensorNAS.Core.Mutate import mutate_int_square

        prev_batch = self.batch_size
        self.batch_size = mutate_int_square(self.batch_size, 1, self.MAX_BATCH_SIZE)
        return "Mutated batch size: {} -> {}".format(prev_batch, self.batch_size)

    def get_keras_model(self, loss, metrics):
        import tensorflow as tf

        inp = tf.keras.Input(shape=self.input_shape)
        try:
            out = self.get_keras_layers(inp)
        except Exception as e:
            raise e

        if out != None:
            try:
                model = tf.keras.Model(inputs=inp, outputs=out)
                model.compile(
                    optimizer=self.opt.get_optimizer(),
                    loss="{}".format(loss),
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

    def train_model(
        self,
        model,
        train_generator=None,
        train_len=None,
        validation_generator=None,
        validation_len=None,
        epochs=1,
        batch_size=1,
        test_name=None,
        model_name=None,
        logger=None,
        verbose=0,
    ):
        import numpy as np
        from Demos import get_global

        callbacks = [ClearMemory()]

        if get_global("use_lrscheduler"):
            from Demos import get_global

            # Do not remove this line
            import TensorNAS.Core.Training as tr

            lrs = eval("tr.{}()".format(get_global("lrscheduler")))

            callbacks += [lrs]

        if get_global("early_stopper"):
            from TensorNAS.Core.Training import get_early_stopper

            callbacks += [get_early_stopper()]

        try:
            if not [
                x
                for x in (
                    train_generator,
                    train_len,
                )
                if x is None
            ]:
                import tensorflow as tf

                model.fit(
                    x=train_generator,
                    batch_size=1,
                    epochs=epochs,
                    steps_per_epoch=train_len // batch_size,
                    callbacks=callbacks,
                    validation_data=validation_generator,
                    validation_batch_size=batch_size,
                    validation_steps=validation_len // batch_size,
                    verbose=verbose,
                )
            else:
                raise Exception("Missing training data")

        except Exception as e:
            print("Error fitting model, {}".format(e))
            return model, np.inf

        params = self.save_model(
            model=model, test_name=test_name, model_name=model_name, logger=logger
        )

        gc.collect()
        return model, params


class AreaUnderCurveBlockArchitecture(BlockArchitecture):
    def evaluate(
        self,
        train_generator=None,
        train_len=None,
        test_generator=None,
        validation_generator=None,
        validation_len=None,
        epochs=1,
        batch_size=1,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        test_name=None,
        model_name=None,
        q_aware=False,
        logger=None,
        verbose=False,
    ):

        model = self.prepare_model(loss=loss, metrics=metrics, q_aware=q_aware)

        if model == None:
            params = math.inf
            accuracy = 0
            return params, accuracy

        model, params = self.train_model(
            model=model,
            train_generator=train_generator,
            train_len=train_len,
            validation_generator=validation_generator,
            validation_len=validation_len,
            epochs=epochs,
            batch_size=batch_size,
            test_name=test_name,
            model_name=model_name,
            logger=logger,
            verbose=verbose,
        )

        try:
            if not test_generator is None:
                import numpy as np
                from tqdm import tqdm
                import sklearn.metrics as metrics

                # true_y = []
                predictions = []
                for td, ty in tqdm(test_generator, total=len(test_generator)):
                    pred = model.predict(td)
                    errors = np.mean(np.mean(np.square(td - pred), axis=1))
                    predictions.append(errors)
                labels = [label for x, label in test_generator]
                auc = metrics.roc_auc_score(labels, predictions)

            else:
                raise Exception("Missing training data")
        except Exception as e:
            auc = 0
            print("Error evaluating model: {}".format(e))

        return params, auc


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
        train_generator=None,
        train_len=None,
        test_generator=None,
        test_len=None,
        validation_generator=None,
        validation_len=None,
        epochs=1,
        batch_size=1,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        test_name=None,
        model_name=None,
        q_aware=False,
        logger=None,
        verbose=False,
    ):

        model = self.prepare_model(loss=loss, metrics=metrics, q_aware=q_aware)

        if model == None:
            params = math.inf
            accuracy = 0
            return params, accuracy

        model, params = self.train_model(
            model=model,
            train_generator=train_generator,
            train_len=train_len,
            validation_generator=validation_generator,
            validation_len=validation_len,
            epochs=epochs,
            batch_size=batch_size,
            test_name=test_name,
            model_name=model_name,
            logger=logger,
            verbose=verbose,
        )

        try:
            if test_generator is not None:
                accuracy = (
                    model.evaluate(
                        x=test_generator,
                        batch_size=batch_size,
                        verbose=verbose,
                        steps=test_len // batch_size,
                    )[1]
                    * 100
                )
            else:
                raise Exception("Missing training data")

        except Exception as e:
            accuracy = 0
            print("Error evaluating model: {}".format(e))

        gc.collect()

        return params, accuracy
