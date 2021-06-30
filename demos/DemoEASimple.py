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
pop_size = 4
gen_count = 2
cxpb = 0.1
mutpb = 0.4
verbose = True


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


def min_max_array(fitnesses, vectors):

    ret = []

    goal_vectors, normalization_vectors = vectors

    for nv, gv in zip(normalization_vectors, goal_vectors):
        ret.append(min_max(fitnesses, nv, gv))

    ret = tuple(ret)
    return ret


def min_max(fitnesses, normalization_vector, goal_vector):

    ret = max(
        (fitnesses[0] - goal_vector[0]) / normalization_vector[0],
        (goal_vector[1] - fitnesses[1]) / normalization_vector[1],
    )
    print("----")
    print(
        "Value Nr. param:"
        + str(fitnesses[0])
        + " Value Acc: "
        + str(fitnesses[1])
        + " obj:"
        + str(ret)
    )
    print(
        "Goal  Nr. param:" + str(goal_vector[0]) + " Goal  Acc: " + str(goal_vector[1])
    )
    print(
        "Norm  Nr. param:"
        + str(normalization_vector[0])
        + " Norm  Acc: "
        + str(normalization_vector[1])
    )
    print("----")
    return ret


def DEAPTestEASimple(
    test, filter_function=None, filter_function_args=None, comment=None
):
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
        filter_function_args=filter_function_args,
    )

    test.ir.save(
        5,
        filter_function.__name__ if filter_function else "no filter func",
        comment=comment,
    )

    return pop, logbook


def setup(objective_weights) -> object:
    from tensornas.tools.DEAPtest import DEAPTest
    from tensornas.core.crossover import crossover_individuals_sp
    from tensornas.tools import GPU as GPU

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


def gen_vectors_variable_goal(g_start, g_stop, g_step, n1, n2):

    if g_start == g_stop:
        goal_vectors = [(g_start, 100)]
    else:
        goal_vectors = [
            (i, 100)
            for i in range(g_start, g_stop + g_step, 1 if not g_step else g_step)
        ]
    normalization_vectors = [(n1, n2) for _ in range(len(goal_vectors))]

    return goal_vectors, normalization_vectors


def gen_vectors_variable_normalization(n_start, n_stop, n_step, g1, g2):

    if n_start == n_stop:
        normalization_vectors = [(n_start, 1)]
    else:
        normalization_vectors = [
            (i, 1) for i in range(n_start, n_stop + n_step, 1 if not n_step else n_step)
        ]
    goal_vectors = [(g1, g2) for _ in range(len(normalization_vectors))]

    return goal_vectors, normalization_vectors


filter_functions_args = [
    gen_vectors_variable_goal(40000, 40000, 0, 1000, 1),
    gen_vectors_variable_goal(40000, 50000, 10000, 1000, 1),
    gen_vectors_variable_goal(40000, 60000, 10000, 1000, 1),
    gen_vectors_variable_normalization(1000, 2000, 1000, 40000, 100),
    gen_vectors_variable_normalization(1000, 5000, 1000, 40000, 100),
]

filter_functions_comments = [
    "G 40000, N 1000,1",
    "G 40000->50000, N 1000,1",
    "G 40000->60000, N 1000,1",
    "G 40000, N 1000->2000,1",
    "G 40000, N 1000->5000, 1",
]

filter_functions = [
    min_max_array,
    min_max_array,
    min_max_array,
    min_max_array,
    min_max_array,
]

filter_function_weights = [
    (-1,),
    (-1, -1),
    (-1, -1, -1),
    (-1, -1),
    (-1, -1, -1, -1, -1),
]

if __name__ == "__main__":
    from tensornas.tools.visualization import plot_hof_pareto

    for ff, w, ff_args, cmnt in zip(
        filter_functions,
        filter_function_weights,
        filter_functions_args,
        filter_functions_comments,
    ):
        test = setup(w)
        name = ff.__name__ if ff else "no filter func"
        print("Start: {}".format(name))

        DEAPTestEASimple(
            test, filter_function=ff, filter_function_args=ff_args, comment=cmnt
        )

        plot_hof_pareto(test.hof, name)
        print("Finished: {}".format(name))

    print("Done")
