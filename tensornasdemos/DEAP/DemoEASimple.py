import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--folder",
    help="Absolute path to folder where interrupted test's output is stored",
    default=None,
)
parser.add_argument("--gen", help="Generation from which the test should resume")

args = parser.parse_args()


def _gen_ba():

    global ba_class, input_tensor_shape, class_count

    return ba_class(input_tensor_shape, class_count)


def _get_block_architecture_class(ba_name):
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

    return classes[1]


def _evaluate_individual(individual, test_name, gen, logger):
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
        model_name="{}/{}".format(gen, individual.index),
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

    existing_generation = None

    if args.folder:
        test_folder = args.folder
        start_gen = args.gen
        existing_generation = test_folder + "/Models/{}".format(start_gen)

        config = LoadConfig(GetConfigFile(directory=test_folder))
    else:
        config_filename = "example"
        config = LoadConfig(GetConfigFile(config_filename=config_filename))

    ba_name = GetBlockArchitecture(config)
    globals()["class_count"] = GetClassCount(config)
    globals()["ba_class"] = _get_block_architecture_class(ba_name)

    if args.folder:
        from pathlib import Path

        test_name = Path(args.folder).name
    else:
        from time import gmtime, strftime

        test_name_prefix = GetOutputPrefix(config)
        test_name = strftime("%d_%m_%Y-%H_%M", gmtime())
        if test_name_prefix:
            test_name = test_name_prefix + "_" + test_name
        test_name += "_" + ba_name

        CopyConfig(config_filename, test_name)

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

    register_DEAP_individual_gen_func(
        creator=creator, toolbox=toolbox, ind_gen_func=_gen_ba
    )

    pop, logbook, test = TestEASimple(
        cxpb=cxpb,
        mutpb=mutpb,
        pop_size=pop_size,
        gen_count=gen_count,
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
        existing_generation=existing_generation,
    )

    print("Done")
