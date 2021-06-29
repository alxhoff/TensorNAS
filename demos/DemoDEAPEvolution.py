import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import multiprocessing

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from deap import base, creator, tools, algorithms

from tensornas.core.individual import Individual
from demos.DemoMNISTInput import *

from math import ceil

### ENABLE GPU ###
gpus = tf.config.experimental.list_physical_devices("GPU")
for device in gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
##################

from tensornas.tools.latexwriter import LatexWriter

lw = LatexWriter()

# Tensorflow parameters
epochs = 1
batch_size = 1
training_size = len(images_train)
step_size = int(ceil((1.0 * training_size) / batch_size)) / 100

optimizer = "adam"
loss = "sparse_categorical_crossentropy"
metrics = ["accuracy"]
pop_size = 12
gen_count = 5

# Functions used for EA demo

# Create a NAS model individual from one of the two demo models
# Creating an iterable that is fed into initIterate


def get_block_architecture():
    from tensornas.blocktemplates.blockarchitectures.ClassificationBlockArchitecture import (
        ClassificationBlockArchitecture,
    )

    """
    This function is responsible for creating and returning the block architecture that an individual should embed
    """
    return ClassificationBlockArchitecture(input_tensor_shape, mnist_class_count)


# Evaluation function for evaluating an individual. This simply calls the evaluate method of the TensorNASModel class
fitnesses = []
fitness_queue = multiprocessing.Queue()


def fitness_recorder():
    global fitnesses
    while True:
        fitnesses.append(fitness_queue.get(block=True))


fitness_task = multiprocessing.Process(target=fitness_recorder)


def log_evaluate_accuracy(fitness):
    from math import log

    ret = 63 * log(1 + fitness, 20) / (1 + 63 * log(1 + fitness, 20))
    return ret


def evaluate_individual(individual):
    ret = individual.evaluate(
        train_data=images_train,
        train_labels=labels_train,
        test_data=images_test,
        test_labels=labels_test,
        epochs=epochs,
        batch_size=batch_size,
        steps=step_size,
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    return ret


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
from tensornas.core.crossover import crossover_individuals_sp

toolbox.register("mate", crossover_individuals_sp)
toolbox.register("mutate", mutate_individual)
toolbox.register("select", tools.selTournamentDCD)

# Statistics
history = tools.History()
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)


def compare_individual(ind1, ind2):

    if ind1.fitness.values == ind2.fitness.values:
        return True

    return False


def main():
    from tensornas.algorithms.eaSimple import eaSimple
    from tensornas.tools.visualization import IndividualRecord

    ir = IndividualRecord()
    pop = toolbox.population(n=pop_size)
    history.update(pop)
    # hof = tools.HallOfFame(1)
    hof = tools.ParetoFront(compare_individual)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max)

    pop, logbook = eaSimple(
        pop,
        toolbox,
        cxpb=0.1,
        mutpb=0.4,
        ngen=gen_count,
        stats=stats,
        halloffame=hof,
        verbose=True,
        individualrecord=ir,
        filter_function=log_evaluate_accuracy,
    )

    ir.save(2)

    return pop, logbook, hof


if __name__ == "__main__":
    pop, log, hof = main()
    best_individual = hof[0]

    gen, std, avg, min_, max_ = log.select("gen", "std", "avg", "min", "max")
    print(log)

    print("### Pareto individuals ###")
    for i in hof.items:
        print(
            "Params: {}, Accuracy: {}".format(i.fitness.values[0], i.fitness.values[1])
        )
        i.print_tree()

    x = [i.fitness.values[0] for i in hof.items]
    y = [i.block_architecture.accuracy for i in hof.items]
    hof_fitness = [i.fitness for i in hof.items]
    # dominated = np.setdiff1d(fitnesses, hof_fitness)

    import matplotlib.backends.backend_agg as agg

    fig = matplotlib.figure.Figure(figsize=(15, 15))
    agg.FigureCanvasAgg(fig)
    plt.figure(figsize=(15, 5))

    divs = 20
    padding = 1.1

    max_x = max(x) * padding
    max_y = max(y) * padding

    x_divs = max_x / divs
    y_divs = max_y / divs

    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(x, y, facecolor=(0.7, 0.7, 0.7), zorder=-1)

    # ax.scatter([i[0] for i in dominated], [i[1] for i in dominated], facecolor="red")
    ax.xscale = "log"

    for item in [(x[i], y[i]) for i in range(1, len(x))]:
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (item[0], 0),
                max_x - item[0],
                item[1],
                lw=0,
                facecolor=(1.0, 0.8, 0.8),
                zorder=-10,
            )
        )

    ax.set_xscale("log")
    ax.set_ylim(bottom=0, top=100)

    fig.savefig("pareto")
