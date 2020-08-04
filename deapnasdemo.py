from tensornas import *
from deap import base, creator, tools, algorithms
import random
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import demomodels
import multiprocessing

# Training MNIST data
(images_train, labels_train), (images_test, labels_test) = keras.datasets.mnist.load_data()
input_shape = images_train.shape
images_train = images_train.reshape(images_train.shape[0], images_train.shape[1], images_train.shape[2], 1)
images_test = images_test.reshape(images_test.shape[0], images_test.shape[1], images_test.shape[2], 1)
input_tensor_shape = (images_test.shape[1], images_test.shape[2], 1)
images_train = images_train.astype('float32')
images_test = images_test.astype('float32')
images_train /= 255
images_test /= 255

# Tensorflow parameters
epochs = 1
batch_size = 100

# Demo hard-coded models
demomodels.generate_demo_model_jsons()
demo_models = demomodels.generate_demo_model_array()
demo_model_count = 2

# Functions used for EA demo

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

# Evaluation function for evaluating an individual. This simply calls the evaluate method of the TensorNASModel class
def evaluate_individual(individual):
    return individual.evaluate(images_train, labels_train, images_test, labels_test, epochs, batch_size)

# Note: please take note of arguments and return forms!
def crossover_individuals(ind1, ind2):
    #TODO
    return ind1,ind2

def mutate_individual(individual):
    #TODO
    return individual,

# We want to minimize param count and maximize accuracy
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))

# Each individual will be an architecture model
creator.create("Individual", TensorNASModel, fitness=creator.FitnessMulti)

toolbox=base.Toolbox()

# pool = multiprocessing.Pool()
# toolbox.register("map", pool.map)

toolbox.register("attr_nas_model_gen", get_demo_model_generator)
toolbox.register("attr_nas_model_itr", get_demo_model_iterator)

toolbox.register("individual_repeat", tools.initRepeat, creator.Individual, toolbox.attr_nas_model_gen, n=6)
toolbox.register("individual_iterate", tools.initIterate, creator.Individual, toolbox.attr_nas_model_itr)

toolbox.register("population", tools.initRepeat, list, toolbox.individual_iterate, n=demo_model_count)

# Genetic operators
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", crossover_individuals)
toolbox.register("mutate", mutate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    pop=toolbox.population(n=3)
    hof=tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=2, stats=stats, halloffame=hof, verbose=True)

    return pop, logbook, hof

if __name__ == "__main__":
    pop, log, hof = main()
    print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))
    hof[0].print()

    gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
    plt.plot(gen, avg, label="average")
    plt.plot(gen, min_, label="minimum")
    plt.plot(gen, max_, label="maximum")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")
    plt.show()