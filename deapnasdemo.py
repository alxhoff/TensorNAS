from tensornas import *
from deap import base
from deap import creator
from deap import tools
import random
import tensorflow as tf

import demomodels

# Training MNIST data
(images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.mnist.load_data()
input_shape = images_train.shape
images_train = images_train.reshape(images_train.shape[0], images_train.shape[1], images_train.shape[2], 1)
images_test = images_test.reshape(images_test.shape[0], images_test.shape[1], images_test.shape[2], 1)
input_tensor_shape = (images_test.shape[1], images_test.shape[2], 1)
images_train = images_train.astype('float32')
images_test = images_test.astype('float32')
images_train /= 255
images_test /= 255

# Demo hard-coded models
demomodels.generate_demo_model_jsons()
demo_models = demomodels.generate_demo_model_array()
demo_model_count = 2

# Create a NAS model individual from one of the two demo models
# Creating an iterable that is fed into initIterate
def get_demo_model_iterator():
    model = demo_models[random.randint(0, demo_model_count - 1)]
    iter = (ModelLayer(model.get(str(layer)).get('name'), model.get(str(layer)).get('args')) for layer in model.keys())
    return iter

# Problem with using generator to provide layers is that the initRepeat method requires 'n' which defines how many
# layers are to be generated. This might vary and as such we should use an iterator where the 'n' of generated
# layers can be modified, this is done using initIterate and get_demo_model_iterator.
def get_demo_model_generator():
    model = demo_models[0]
    for layer in model.keys():
        target_layer = model.get(str(layer))
        yield ModelLayer(target_layer.get('name'), target_layer.get('args'))

# We want to minimize param count and maximize accuracy
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))

# Each individual will be an architecture model
creator.create("Individual", TensorNASModel, fitness=creator.FitnessMulti)

toolbox=base.Toolbox()

toolbox.register("attr_nas_model_gen", get_demo_model_generator)
toolbox.register("attr_nas_model_itr", get_demo_model_iterator)

toolbox.register("individual_repeat", tools.initRepeat, creator.Individual, toolbox.attr_nas_model_gen, n=6)
toolbox.register("individual_iterate", tools.initIterate, creator.Individual, toolbox.attr_nas_model_itr)

toolbox.register("population", tools.initRepeat, list, toolbox.individual_iterate, n=demo_model_count)

ind = toolbox.individual_iterate()
pop=toolbox.population(n=3)

epochs = 10
batch_size = 1000

pop[0].train(data=images_train, labels=labels_train, epochs=epochs, batch_size=batch_size)
print("First population has accuracy of {}".format(pop[0].evaluate))
