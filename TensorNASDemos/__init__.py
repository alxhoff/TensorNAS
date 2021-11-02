def get_config(args=None):
    from time import gmtime, strftime
    from TensorNAS.Tools.ConfigParse import (
        LoadConfig,
        GetConfigFile,
        GetBlockArchitecture,
    )
    from TensorNAS.Tools.ConfigParse import (
        GetOutputPrefix,
        CopyConfig,
    )

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

    if not get_global("test_name"):
        test_name_prefix = GetOutputPrefix(config)
        set_global("test_name", strftime("%d_%m_%Y-%H_%M", gmtime()))
        if test_name_prefix:
            set_global("test_name", test_name_prefix + "_" + get_global("test_name"))
        set_global("test_name", get_global("test_name") + "_" + get_global("ba_name"))
        CopyConfig(config_filename, get_global("test_name"))

    return config


def gen_ba():
    global ba_mod, input_tensor_shape, class_count, batch_size, optimizer

    return ba_mod.Block(input_tensor_shape, class_count, batch_size, optimizer)


def evaluate_individual(individual, test_name, gen, logger):
    global epochs, batch_size, loss, metrics, images_train, images_test, labels_train, labels_test, train_generator
    global val_generator, save_individuals, use_gpu, q_aware, steps_per_epoch

    param_count, accuracy = individual.evaluate(
        train_data=images_train,
        train_labels=labels_train,
        test_data=images_test,
        test_labels=labels_test,
        train_generator=train_generator,
        val_generator=val_generator,
        epochs=epochs,
        batch_size=batch_size,
        loss=loss,
        metrics=metrics,
        test_name=test_name,
        model_name="{}/{}".format(gen, individual.index),
        use_GPU=use_gpu,
        q_aware=q_aware,
        logger=logger,
        steps_per_epoch=steps_per_epoch,
    )

    return param_count, accuracy


def mutate_individual(individual):
    return (individual.mutate(),)


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
    globals()["filter_function"] = GetFilterFunction(config)
    globals()["filter_function_args"] = GetFilterFunctionArgs(config)
    globals()["weights"] = GetWeights(config)
    globals()["comments"] = GetFigureTitle(config)


def set_test_train_data(
    train_data=None,
    train_labels=None,
    test_data=None,
    test_labels=None,
    train_generator=None,
    val_generator=None,
    input_tensor_shape=None,
    training_sample_size=None,
    test_sample_size=None,
):

    globals()["train_generator"] = train_generator
    globals()["val_generator"] = val_generator
    steps = 0

    if training_sample_size and train_data and train_labels:
        steps = 1
        globals()["images_train"] = train_data[:training_sample_size]
        globals()["labels_train"] = train_labels[:training_sample_size]
    else:
        globals()["images_train"] = train_data
        globals()["labels_train"] = train_labels

    if test_sample_size and test_data and test_labels:
        globals()["images_test"] = test_data[:test_sample_size]
        globals()["labels_test"] = test_labels[:test_sample_size]
    else:
        globals()["images_test"] = test_data
        globals()["labels_test"] = test_labels

    if train_generator:
        from TensorNASDemos import get_global

        steps = training_sample_size / get_global("batch_size")

    globals()["input_tensor_shape"] = input_tensor_shape
    globals()["steps_per_epoch"] = steps


def load_tensorflow_params_from_config(config):
    from TensorNAS.Tools.ConfigParse import (
        GetTFEpochs,
        GetTFBatchSize,
        GetTFOptimizer,
        GetTFLoss,
        GetTFMetrics,
        GetTFQuantizationAware,
        GetTrainingSampleSize,
        GetTestSampleSize,
    )

    globals()["epochs"] = GetTFEpochs(config)
    globals()["batch_size"] = GetTFBatchSize(config)
    globals()["optimizer"] = GetTFOptimizer(config)
    globals()["loss"] = GetTFLoss(config)
    globals()["metrics"] = GetTFMetrics(config)
    globals()["q_aware"] = GetTFQuantizationAware(config)
    globals()["training_sample_size"] = GetTrainingSampleSize(config)
    globals()["test_sample_size"] = GetTestSampleSize(config)


def get_global(var_name):
    try:
        return globals()[var_name]
    except:
        return None


def set_global(var_name, val):

    globals()[var_name] = val