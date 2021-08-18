def _get_block_architecture():
    global input_tensor_shape, ba_name, class_count

    from importlib import import_module

    ba_module = import_module(
        "tensornas.blocktemplates.blockarchitectures.{}".format(ba_name)
    )

    import inspect

    classes = inspect.getmembers(
        ba_module, lambda member: inspect.isclass(member) and member.__name__ == ba_name
    )

    assert len(classes) == 1

    classes = classes[0]

    assert classes[0] == ba_name

    classes = classes[1]

    return classes(input_tensor_shape, class_count)


def _evaluate_individual(individual, test_name, gen, ind_num, logger):
    global epochs, batch_size, optimizer, loss, metrics, images_train, images_test, labels_train, labels_test, save_individuals, use_gpu, q_aware

    ret = individual.evaluate(
        train_data=images_train,
        train_labels=labels_train,
        test_data=images_test,
        test_labels=labels_test,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        test_name=test_name,
        model_name="{}/{}".format(gen, ind_num),
        use_GPU=use_gpu,
        q_aware=q_aware,
        logger=logger,
    )
    param_count = ret[0]
    accuracy = ret[1]
    return param_count, accuracy


def _mutate_individual(individual):
    return (individual.mutate(),)


if __name__ == "__main__":

    from tensornas.tools.configparse import *

    config_filename = "example"
    config = LoadConfig(config_filename)

    from time import gmtime, strftime

    test_name_prefix = GetOutputPrefix(config)
    test_name = strftime("%d_%m_%Y-%H_%M", gmtime())
    if test_name_prefix:
        test_name = test_name_prefix + "_" + test_name

    CopyConfig(config_filename, test_name)

    globals()["ba_name"] = GetBlockArchitecture(config)
    globals()["class_count"] = GetClassCount(config)

    training_sample_size = GetTrainingSampleSize(config)
    test_sample_size = GetTestSampleSize(config)
    globals()["epochs"] = GetTFEpochs(config)
    globals()["batch_size"] = GetTFBatchSize(config)
    globals()["optimizer"] = GetTFOptimizer(config)
    globals()["loss"] = GetTFLoss(config)
    globals()["metrics"] = GetTFMetrics(config)
    globals()["q_aware"] = GetQuantizationAware(config)

    pop_size = GetPopulationSize(config)
    gen_count = GetGenerationCount(config)
    cxpb = GetCrossoverProbability(config)
    mutpb = GetMutationProbability(config)
    verbose = GetVerbose(config)
    multithreaded = GetMultithreaded(config)

    thread_count = GetThreadCount(config)
    log = GetLog(config)

    globals()["use_gpu"] = GetGPU(config)
    globals()["save_individuals"] = GetSaveIndividual(config)
    generation_gap = GetGenerationGap(config)
    generation_save_interval = GetGenerationSaveInterval(config)

    from tensornasdemos.Datasets.MNIST import GetData

    images_test, images_train, labels_test, labels_train, input_tensor_shape = GetData()
    images_train = images_train[:training_sample_size]
    labels_train = labels_train[:training_sample_size]
    images_test = images_test[:test_sample_size]
    labels_test = labels_test[:test_sample_size]
    globals()["images_test"] = images_test
    globals()["images_train"] = images_train
    globals()["labels_test"] = labels_test
    globals()["labels_train"] = labels_train
    globals()["input_tensor_shape"] = input_tensor_shape

    filter_function = GetFilterFunction(config)
    filter_function_args = GetFilterFunctionArgs(config)
    weights = GetWeights(config)
    comments = GetFigureTitle(config)

    from tensornas.algorithms.EASimple import TestEASimple
    from tensornas.core.crossover import crossover_individuals_sp

    # import re

    # gen_functions = [
    #     func
    #     for func in filter(callable, globals().values())
    #     if re.search(r"^_gen", func.__name__)
    # ]

    from importlib import import_module

    creator = import_module("deap.creator")
    toolbox = import_module("deap.base").Toolbox()

    from tensornas.tools.DEAPtest import setup_DEAP, register_DEAP_individual_gen_func

    setup_DEAP(
        creator=creator,
        toolbox=toolbox,
        objective_weights=weights,
        multithreaded=multithreaded,
        thread_count=thread_count,
    )

    # for gen_func in gen_functions:
    register_DEAP_individual_gen_func(
        creator=creator, toolbox=toolbox, ind_gen_func=_get_block_architecture
    )

    pop, logbook, test = TestEASimple(
        cxpb=cxpb,
        mutpb=mutpb,
        pop_size=pop_size,
        gen_count=gen_count,
        gen_individual=_get_block_architecture,
        evaluate_individual=_evaluate_individual,
        crossover_individual=crossover_individuals_sp,
        mutate_individual=_mutate_individual,
        toolbox=toolbox,
        test_name=test_name,
        verbose=verbose,
        filter_function=filter_function,
        filter_function_args=filter_function_args,
        save_individuals=save_individuals,
        generation_gap=generation_gap,
        generation_save=generation_save_interval,
        comment=comments,
        multithreaded=multithreaded,
        log=log,
    )

    print("Done")
