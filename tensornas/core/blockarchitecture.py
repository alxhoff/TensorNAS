import numpy as np
import tensorflow as tf
import os
import pickle
import datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.sys.path.append("/Users/priyadalal/Desktop/priya/TensorNAS")
from tensornas.core.block import Block
from tensornas.core.layer import NetworkLayer
from ann_visualizer.visualize import ann_viz	#import pydot
import pydotplus	#from ann_visualizer.visualize import ann_viz
from pydotplus import graphviz	#import pydotplus
from keras_sequential_ascii import keras2ascii	#from pydotplus import graphviz

#from keras_sequential_ascii import
class ReportValidationStatus(tf.keras.callbacks.Callback):

    def on_test_batch_begin(self, batch, logs=None):
        print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

    def on_test_batch_end(self, batch, logs=None):
        print('Evaluating: batch {} ends at {}'.format(batch,  datetime.datetime.now().time()))

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
            #todo: change to VWW train iterator if needed
            #train_it,
            #test_it,
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
            #todo: extra code to save model.summary
            #
            folder='/Users/priyadalal/Desktop/priya/TensorNAS/Results/Model_summary'
            def myprint(s):
                with open(f'{folder}/modelsummary.txt','w+') as f:
                    print(s, file=f)
            model.summary(print_fn=myprint)
            if filename:
                from tensornas.core.util import save_model
                save_model(model, filename)

            if batch_size == -1:
                #todo: change to VWW train iterator if needed
                #model.fit_generator(train_it,steps_per_epoch=steps)

                
                model.fit(
                    x=train_data,
                    y=train_labels,
                    epochs=epochs,
                    steps_per_epoch=steps,
                    verbose=1,
                )

            else:
                #todo: change to VWW train iterator if needed
                #model.fit_generator(train_it,steps_per_epoch=steps)

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
            return [np.inf, 0]
        params = int(
            np.sum([tf.keras.backend.count_params(p) for p in model.trainable_weights])
        ) + int(
            np.sum(
                [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
            )
        )
        if params == 0:
            params = np.inf
        #todo: remove prints
        print('87')
        #todo: change to VWW train iterator if needed
        #accuracy=(model.evaluate_generator(test_it,32))[1]*100
        print('89')

        accuracy = model.evaluate(test_data, test_labels)[1] * 100
        print('accuracy ',accuracy)
        return params, accuracy