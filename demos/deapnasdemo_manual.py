import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import multiprocessing

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from deap import base, creator, tools, algorithms

from tensornas.core.individual import Individual
from demos.mnistdemoinput import *


# Tensorflow parameters
epochs = 1
batch_size = 600
steps = 5
optimizer = "adam"
loss = "sparse_categorical_crossentropy"
metrics = ["accuracy"]
pop_size = 10


# Functions used for EA demo

# Create a NAS model individual from one of the two demo models
# Creating an iterable that is fed into initIterate


def get_block_architecture():
    from tensornas.blocktemplates.blockarchitectures.classificationblockarchitectures import (
        ClassificationBlockArchitecture,
    )

    """
    This function is responsible for creating and returning the block architecture that an individual shuld embed
    """
    return ClassificationBlockArchitecture(input_tensor_shape, mnist_class_count)


# Evaluation function for evaluating an individual. This simply calls the evaluate method of the TensorNASModel class
def evaluate_individual(individual):
    return individual.evaluate(
        train_data=images_train,
        train_labels=labels_train,
        test_data=images_test,
        test_labels=labels_test,
        epochs=epochs,
        batch_size=batch_size,
        steps=steps,
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )


# Note: please take note of arguments and return forms!
def crossover_individuals(ind1, ind2):
    from copy import deepcopy
    from tensornas.core.crossover import crossover_single_point

    ind3, ind4 = deepcopy(ind1), deepcopy(ind2)
    ind3.block_architecture, ind4.block_architecture = crossover_single_point(
        ind3.block_architecture, ind4.block_architecture
    )
    return ind3, ind4


def mutate_individual(individual):
    from copy import deepcopy

    ind2 = deepcopy(individual)
    ind2.mutate
    return (ind2,)


def pareto_dominance(ind1, ind2):
    return tools.emo.isDominated(ind1.fitness.values, ind2.fitness.values)


def print_ind(ind):
    print(ind.block_architecture.get_ascii_tree())


# We want to minimize param count and maximize accuracy
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))

# Each individual will be an architecture model
creator.create("Individual", Individual, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

### Multithreading ###
pool = multiprocessing.Pool()
toolbox.register("map", pool.map)
######

toolbox.register("get_block_architecture", get_block_architecture)
toolbox.register(
    "individual",
    tools.initRepeat,
    creator.Individual,
    toolbox.get_block_architecture,
    n=1,
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=pop_size)

# Genetic operators
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", crossover_individuals)
toolbox.register("mutate", mutate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("print", print_ind)

# Statistics
history = tools.History()
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)


def main():
    ind1 = toolbox.individual()
    ind2 = toolbox.individual()

    (ind3,) = toolbox.mutate(ind1)

    ind4, ind5 = toolbox.mate(ind1, ind2)

    print("####################################################################")
    ind1.print_tree()
    print("####################################################################")
    ind2.print_tree()
    print("####################################################################")
    ind3.print_tree()
    print("####################################################################")
    ind4.print_tree()
    print("####################################################################")
    ind5.print_tree()
    print("####################################################################")

    print("Done")


if __name__ == "__main__":
    main()
