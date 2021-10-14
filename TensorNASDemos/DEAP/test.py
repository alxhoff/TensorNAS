def gen_ba():
    global ba_mod, input_tensor_shape, class_count, batch_size, optimizer

    return ba_mod.Block(input_tensor_shape, class_count, batch_size, optimizer)


def evaluate_individual(individual, test_name, gen, logger):
    global epochs, batch_size, loss, metrics, images_train, images_test, labels_train, labels_test, save_individuals, use_gpu, q_aware

    param_count, accuracy = individual.evaluate(
        train_data=images_train,
        train_labels=labels_train,
        test_data=images_test,
        test_labels=labels_test,
        epochs=epochs,
        batch_size=batch_size,
        loss=loss,
        metrics=metrics,
        test_name=test_name,
        model_name="{}/{}".format(gen, individual.index),
        use_GPU=use_gpu,
        q_aware=q_aware,
        logger=logger,
    )

    return param_count, accuracy


def mutate_individual(individual):
    return (individual.mutate(),)


def get_config(args=None):
    from time import gmtime, strftime
    from TensorNAS.Tools.ConfigParse import (
        GetBlockArchitecture,
        LoadConfig,
        GetConfigFile,
        GetOutputPrefix,
        CopyConfig,
    )

    globals()["existing_generation"] = None
    globals()["start_gen"] = 0
    globals()["test_name"] = None
    config = None
    config_filename = "example"

    if args:
        if args.folder:
            from pathlib import Path

            globals()["test_name"] = Path(args.folder).name
            globals()["existing_generation"] = args.folder + "/Models/{}".format(
                args.gen
            )

            config = LoadConfig(GetConfigFile(directory=args.folder))
        if args.config:
            config_filename = args.config
            config = LoadConfig(GetConfigFile(config_filename=args.config))

    else:
        config = LoadConfig(GetConfigFile(config_filename=config_filename))

    globals()["ba_name"] = GetBlockArchitecture(config)

    if not globals()["test_name"]:

        test_name_prefix = GetOutputPrefix(config)
        globals()["test_name"] = strftime("%d_%m_%Y-%H_%M", gmtime())
        if test_name_prefix:
            globals()["test_name"] = test_name_prefix + "_" + globals()["test_name"]
        globals()["test_name"] += "_" + get_global("ba_name")
        CopyConfig(config_filename, globals()["test_name"])

    return config


def load_globals_from_config(config):
    from TensorNAS.Tools.ConfigParse import (
        GetBlockArchitecture,
        GetClassCount,
        GetLog,
        GetVerbose,
        GetMultithreaded,
        GetThreadCount,
        GetGPU,
        GetSaveIndividual,
        GetGenerationGap,
        GetGenerationSaveInterval,
        GetFilterFunction,
        GetFilterFunctionArgs,
        GetWeights,
        GetFigureTitle,
    )
    from TensorNAS.Tools.JSONImportExport import GetBlockMod

    globals()["ba_name"] = GetBlockArchitecture(config)
    globals()["class_count"] = GetClassCount(config)
    globals()["ba_mod"] = GetBlockMod(globals()["ba_name"])
    globals()["log"] = GetLog(config)
    globals()["verbose"] = GetVerbose(config)
    globals()["multithreaded"] = GetMultithreaded(config)
    globals()["thread_count"] = GetThreadCount(config)
    globals()["use_gpu"] = GetGPU(config)
    globals()["save_individuals"] = GetSaveIndividual(config)
    globals()["generation_gap"] = GetGenerationGap(config)
    globals()["generation_save_interval"] = GetGenerationSaveInterval(config)
    globals()["filter_function"] = GetFilterFunction(config)
    globals()["filter_function_args"] = GetFilterFunctionArgs(config)
    globals()["weights"] = GetWeights(config)
    globals()["comments"] = GetFigureTitle(config)


def set_test_train_data(
    train_data,
    train_labels,
    test_data,
    test_labels,
    input_tensor_shape,
    training_sample_size=None,
    test_sample_size=None,
):

    if training_sample_size:
        globals()["images_train"] = train_data[:training_sample_size]
        globals()["labels_train"] = train_labels[:training_sample_size]
    else:
        globals()["images_train"] = train_data
        globals()["labels_train"] = train_labels

    if test_sample_size:
        globals()["images_test"] = test_data[:test_sample_size]
        globals()["labels_test"] = test_labels[:test_sample_size]
    else:
        globals()["images_test"] = test_data
        globals()["labels_test"] = test_labels

    globals()["input_tensor_shape"] = input_tensor_shape


def load_genetic_params_from_config(config):

    from TensorNAS.Tools.ConfigParse import (
        GetPopulationSize,
        GetGenerationCount,
        GetCrossoverProbability,
        GetMutationProbability,
        GetTrainingSampleSize,
        GetTestSampleSize,
    )

    globals()["pop_size"] = GetPopulationSize(config)
    globals()["gen_count"] = GetGenerationCount(config)
    globals()["cxpb"] = GetCrossoverProbability(config)
    globals()["mutpb"] = GetMutationProbability(config)
    globals()["training_sample_size"] = GetTrainingSampleSize(config)
    globals()["test_sample_size"] = GetTestSampleSize(config)


def load_tensorflow_params_from_config(config):
    from TensorNAS.Tools.ConfigParse import (
        GetTFEpochs,
        GetTFBatchSize,
        GetTFOptimizer,
        GetTFLoss,
        GetTFMetrics,
        GetTFQuantizationAware,
    )

    globals()["epochs"] = GetTFEpochs(config)
    globals()["batch_size"] = GetTFBatchSize(config)
    globals()["optimizer"] = GetTFOptimizer(config)
    globals()["loss"] = GetTFLoss(config)
    globals()["metrics"] = GetTFMetrics(config)
    globals()["q_aware"] = GetTFQuantizationAware(config)


def get_global(var_name):

    return globals()[var_name]


def run_deap_test(evaluate_individual, crossover, mutate):
    from importlib import import_module
    from TensorNAS.Tools.DEAPtest import setup_DEAP, register_DEAP_individual_gen_func
    from TensorNAS.Algorithms.EASimple import TestEASimple

    creator = import_module("deap.creator")
    toolbox = import_module("deap.base").Toolbox()

    setup_DEAP(
        creator=creator,
        toolbox=toolbox,
        objective_weights=get_global("weights"),
        multithreaded=get_global("multithreaded"),
        thread_count=get_global("thread_count"),
    )

    register_DEAP_individual_gen_func(
        creator=creator, toolbox=toolbox, ind_gen_func=gen_ba
    )

    pop, logbook, test = TestEASimple(
        cxpb=get_global("cxpb"),
        mutpb=get_global("mutpb"),
        pop_size=get_global("pop_size"),
        gen_count=get_global("gen_count"),
        evaluate_individual=evaluate_individual,
        crossover_individual=crossover,
        mutate_individual=mutate,
        toolbox=toolbox,
        test_name=get_global("test_name"),
        verbose=get_global("verbose"),
        filter_function=get_global("filter_function"),
        filter_function_args=get_global("filter_function_args"),
        save_individuals=get_global("save_individuals"),
        generation_gap=get_global("generation_gap"),
        generation_save=get_global("generation_save_interval"),
        comment=get_global("comments"),
        multithreaded=get_global("multithreaded"),
        log=get_global("log"),
        start_gen=get_global("start_gen"),
    )

    return pop, logbook, test
