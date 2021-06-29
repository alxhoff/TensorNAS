from DemoMNISTInput import *
from math import ceil

# Tensorflow parameters
epochs = 1
batch_size = 1
training_size = len(images_train)
step_size = int(ceil((1.0 * training_size) / batch_size)) / 100

optimizer = "adam"
loss = "sparse_categorical_crossentropy"
metrics = ["accuracy"]
pop_size = 40
gen_count = 10
cxpb = 0.1
mutpb = 0.4
verbose = True

normalization_vector = [1000, 1]
goal_vector = [20000, 100]


def _gen_block_architecture():
    from tensornas.blocktemplates.blockarchitectures.ClassificationBlockArchitecture import (
        ClassificationBlockArchitecture,
    )

    """
    This function is responsible for creating and returning the block architecture that an individual should embed
    """
    return ClassificationBlockArchitecture(input_tensor_shape, mnist_class_count)


def _evaluate_individual(individual):
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
    param_count = ret[0]
    accuracy = ret[1]
    return param_count, accuracy


def _mutate_individual(individual):
    return (individual.mutate(),)


def _compare_individual(ind1, ind2):

    if ind1.fitness.values == ind2.fitness.values:
        return True

    return False


def log_evaluate_accuracy(fitnesses):
    """Tales a tuple fitness of param count and accuracy, must return the same."""
    from math import log

    acc = fitnesses[1]
    param_count = fitnesses[0] * log(acc + 1, 2)
    return acc, param_count


def step_filter(fitness):

    if fitness[1] > 50:
        return fitness
    else:
        return fitness[0], 0


def min_max(fitnesses):

    ret = max(
        (fitnesses[0] - goal_vector[0]) / normalization_vector[0],
        (goal_vector[1] - fitnesses[1]) / normalization_vector[1],
    )
    return (ret,)


def DEAPTestEASimple(test, filter_function=None):
    from tensornas.algorithms.eaSimple import eaSimple

    pop, logbook = eaSimple(
        population=test.pop,
        toolbox=test.toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=gen_count,
        stats=test.stats,
        halloffame=test.hof,
        verbose=verbose,
        individualrecord=test.ir,
        filter_function=filter_function,
    )

    test.ir.show(1, filter_function.__name__ if filter_function else "no filter func")

    return pop, logbook


def setup(objective_weights) -> object:
    from tensornas.tools.DEAPtest import DEAPTest
    from tensornas.core.crossover import crossover_individuals_sp
    import tensornas.tools.GPU as GPU

    GPU.Enable()

    test = DEAPTest(
        pop_size=pop_size,
        gen_count=gen_count,
        f_gen_individual=_gen_block_architecture,
        objective_weights=objective_weights,
        multithreaded=True,
    )

    test.set_evaluate(func=_evaluate_individual)
    test.set_mate(func=crossover_individuals_sp)
    test.set_mutate(func=_mutate_individual)
    from deap import tools

    test.set_select(func=tools.selTournamentDCD)

    return test


# filter_functions = [log_evaluate_accuracy, step_filter, min_max, None]
filter_functions = [None]

# filter_function_weights = [(-1, 1), (-1, 1), (-1, 1), (-1,)]
filter_function_weights = [(-1, 1)]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tensornas.tools.visualization import plot_hof_pareto

    for filter_function, weights in zip(filter_functions, filter_function_weights):
        test = setup(weights)
        DEAPTestEASimple(test, filter_function=filter_function)
        name = filter_function.__name__ if filter_function else "no filter func"
        plot_hof_pareto(test.hof, name)
        print("Finished: {}".format(name))

    plt.show()
    print("Done")
