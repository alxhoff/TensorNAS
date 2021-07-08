from math import ceil


def _gen_block_architecture():

    input_tensor_shape
    mnist_class_count = 10

    from tensornas.blocktemplates.blockarchitectures.ClassificationBlockArchitecture import (
        ClassificationBlockArchitecture,
    )

    """
    This function is responsible for creating and returning the block architecture that an individual should embed
    """
    return ClassificationBlockArchitecture(input_tensor_shape, mnist_class_count)


def _evaluate_individual(individual):
    global epochs, batch_size, optimizer, loss, metrics, images_train, images_test, labels_train, labels_test

    training_size = len(images_train)
    step_size = int(ceil((1.0 * training_size) / batch_size)) / 100

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
        use_GPU=True,
    )
    param_count = ret[0]
    accuracy = ret[1]
    return param_count, accuracy


def _mutate_individual(individual):
    return (individual.mutate(),)


if __name__ == "__main__":
    from demos.Datasets.MNIST import GetData

    images_test, images_train, labels_test, labels_train, input_tensor_shape = GetData()

    from tensornas.tools.configparse import *

    config = LoadConfig("example")

    globals()["epochs"] = GetTFEpochs(config)
    globals()["batch_size"] = GetTFBatchSize(config)
    globals()["optimizer"] = GetTFOptimizer(config)
    globals()["loss"] = GetTFLoss(config)
    globals()["metrics"] = GetTFMetrics(config)

    pop_size = GetPopulationSize(config)
    gen_count = GetGenerationCount(config)
    cxpb = GetCrossoverProbability(config)
    mutpb = GetMutationProbability(config)
    verbose = GetVerbose(config)

    globals()["images_test"] = images_test
    globals()["images_train"] = images_train
    globals()["labels_test"] = labels_test
    globals()["labels_train"] = labels_train
    globals()["input_tensor_shape"] = input_tensor_shape

    filter_function = GetFilterFunction(config)
    filter_function_args = GetFilterFunctionArgs(config)
    weights = GetWeights(config)
    comments = GetFigureTitle(config)

    name = filter_function.__name__ if filter_function else "no filter func"

    from tensornas.algorithms.EASimple import TestEASimple
    from tensornas.core.crossover import crossover_individuals_sp

    pop, logbook, test = TestEASimple(
        cxpb=cxpb,
        mutpb=mutpb,
        pop_size=pop_size,
        gen_count=gen_count,
        gen_individual=_gen_block_architecture,
        evaluate_individual=_evaluate_individual,
        crossover_individual=crossover_individuals_sp,
        mutate_individual=_mutate_individual,
        objective_weights=weights,
        verbose=verbose,
        filter_function=filter_function,
        filter_function_args=filter_function_args,
        comment=comments,
    )

    from tensornas.tools.visualization import plot_hof_pareto

    plot_hof_pareto(test.hof, name)

    print("Done")
