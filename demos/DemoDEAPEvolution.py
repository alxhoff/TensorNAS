import os
import pickle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.sys.path.append("/Users/priyadalal/Desktop/priya/TensorNAS")
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
#gpus = tf.config.experimental.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(gpus[0], True)
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
#change divs pop_size,gen_count
pop_size = 10
gen_count = 10

# Functions used for EA demo

# Create a NAS model individual from one of the two demo models
# Creating an iterable that is fed into initIterate


def get_block_architecture():
    #from tensornas.blocktemplates.blockarchitectures.ClassificationBlockArchitecture import (
    #    ClassificationBlockArchitecture,
    #)

    from tensornas.blocktemplates.blockarchitectures.MobileNetBlockArchitecture import (
    MobileNetBlockArchitecture,
    )
    #from tensornas.blocktemplates.blockarchitectures.EffNetBlockArchitecture import (
    #EffNetBlockArchitecture,
    #)
    #from tensornas.blocktemplates.blockarchitectures.InceptionNetArchitecture import (
    #InceptionNetBlockArchitecture,
    #)
    #from tensornas.blocktemplates.blockarchitectures.ShuffleNetBlockArchitecture import (
    #ShuffleNetBlockArchitecture,
    #)
    #from tensornas.blocktemplates.blockarchitectures.GhostNetBlockArchitecture import (
    #GhostNetBlockArchitecture,
    #)
    #from tensornas.blocktemplates.blockarchitectures.ResNetBlockArchitecture import (
    #ResNetBlockArchitecture,
    #)
    #from tensornas.blocktemplates.blockarchitectures.SqueezeNetBlockArchitecture import (
    #SqueezeNetBlockArchitecture,
    #)
    #from tensornas.blocktemplates.blockarchitectures.MixedArchitecture import(
    #    MixedBlockArchitecture,
    #)

    """
    This function is responsible for creating and returning the block architecture that an individual shuld embed
    """
    #return ClassificationBlockArchitecture(input_tensor_shape, mnist_class_count)
    return MobileNetBlockArchitecture(input_tensor_shape,mnist_class_count)
    #return EffNetBlockArchitecture(input_tensor_shape,mnist_class_count)
    #return ShuffleNetBlockArchitecture(input_tensor_shape,mnist_class_count)
    #return GhostNetBlockArchitecture(input_tensor_shape,mnist_class_count)
    #return SqueezeNetBlockArchitecture(input_tensor_shape,mnist_class_count)
    #return ResNetBlockArchitecture(input_tensor_shape,mnist_class_count)
    #return InceptionNetBlockArchitecture(input_tensor_shape,mnist_class_count)
    #return MixedBlockArchitecture(input_tensor_shape,mnist_class_count)
# Evaluation function for evaluating an individual. This simply calls the evaluate method of the TensorNASModel class
fitnesses = []
fitness_queue = multiprocessing.Queue()



def fitness_recorder():
    global fitnesses
    while True:
        fitnesses.append(fitness_queue.get(block=True))


fitness_task = multiprocessing.Process(target=fitness_recorder)


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
    fitness_queue.put(ret, block=True)
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
#pool = multiprocessing.Pool()
#toolbox.register("map", pool.map)
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
toolbox.register("select", tools.selTournament, tournsize=3)

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
        cxpb=0.05,
        mutpb=0.2,
        ngen=gen_count,
        stats=stats,
        halloffame=hof,
        verbose=True,
        individualrecord=ir,
    )

    ir.show(2)

    return pop, logbook, hof


if __name__ == "__main__":

    folder='/Users/priyadalal/Desktop/priya/TensorNAS/Results'
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
    y = [i.fitness.values[1] for i in hof.items]
    hof_fitness = [i.fitness for i in hof.items]
    dominated = np.setdiff1d(fitnesses, hof_fitness)

    import matplotlib.backends.backend_agg as agg

    fig = matplotlib.figure.Figure(figsize=(15, 15))
    agg.FigureCanvasAgg(fig)
    plt.figure(figsize=(15, 5))

    divs = 10
    padding = 1.1

    max_x = max(x) * padding
    max_y = max(y) * padding

    x_divs = max_x / divs
    y_divs = max_y / divs

    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(x, y, facecolor=(0.7, 0.7, 0.7), zorder=-1)
    i=0
    while 1:
        if os.path.isfile(f'{folder}\\Dominated_{i}.csv'):
            i+=1
        else:
            with open(f'{folder}\\Dominated_{i}.csv','w') as f:
                for row in dominated:
                    np.savetxt(f, row)
                #f.write(dominated)
            break


    ax.scatter([i[0] for i in dominated], [i[1] for i in dominated], facecolor="red")
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
    while 1:
        if os.path.isfile(f'{folder}\\Pareto_{i}'):
            i+=1
        else:
            with open(f'{folder}\\Pareto_{i}','w') as f:
                fig.savefig(f'{folder}\\Pareto_{i}')
            break
    fig.savefig(f"{folder}\\pareto")
