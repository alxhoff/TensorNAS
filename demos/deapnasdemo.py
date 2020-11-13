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
    from tensornas.core.crossover import crossover_single_point

    ind3, ind4 = crossover_single_point(ind1, ind2)
    return ind3, ind4


def mutate_individual(individual):
    return (individual.mutate(),)


def pareto_dominance(ind1, ind2):
    return tools.emo.isDominated(ind1.fitness.values, ind2.fitness.values)


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

# Statistics
history = tools.History()
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)


def main():
    pop = toolbox.population(n=4)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=2,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    return pop, logbook, hof


if __name__ == "__main__":
    pop, log, hof = main()
    best_individual = hof[0]
    print(
        "Best individual is: %s\nwith fitness: %s"
        % (best_individual, best_individual.fitness)
    )
    best_individual.print()

    gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(gen, avg, label="average")
    plt.plot(gen, min_, label="minimum")
    plt.plot(gen, max_, label="maximum")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")

    # Pareto
    dominated = [
        ind
        for ind in history.genealogy_history.values()
        if pareto_dominance(best_individual, ind)
    ]
    dominators = [
        ind
        for ind in history.genealogy_history.values()
        if pareto_dominance(ind, best_individual)
    ]
    others = [
        ind
        for ind in history.genealogy_history.values()
        if not ind in dominated and not ind in dominators
    ]

    plt.subplot(1, 3, 2)
    for ind in dominators:
        plt.plot(ind.fitness.values[0], ind.fitness.values[1], "r.", alpha=0.7)
    for ind in dominated:
        plt.plot(ind.fitness.values[0], ind.fitness.values[1], "g.", alpha=0.7)
    if len(others):
        for ind in others:
            plt.plot(
                ind.fitness.values[0], ind.fitness.values[1], "k.", alpha=0.7, ms=3
            )
    plt.plot(
        best_individual.fitness.values[0], best_individual.fitness.values[1], "bo", ms=6
    )
    plt.xlabel("$f_1(\mathbf{x})$")
    plt.ylabel("$f_2(\mathbf{x})$")
    plt.xlim((0.5, 3.6))
    plt.ylim((0.5, 3.6))
    plt.title("Objective space")
    plt.tight_layout()

    non_dom = tools.sortNondominated(
        history.genealogy_history.values(),
        k=len(history.genealogy_history.values()),
        first_front_only=True,
    )[0]

    plt.subplot(1, 3, 3)
    for ind in history.genealogy_history.values():
        plt.plot(ind.fitness.values[0], ind.fitness.values[1], "k.", ms=3, alpha=0.5)
    for ind in non_dom:
        plt.plot(ind.fitness.values[0], ind.fitness.values[1], "bo", alpha=0.74, ms=5)
    plt.title("Pareto-optimal front")

    plt.draw()
    # plt.show()

    print("Wait here")
