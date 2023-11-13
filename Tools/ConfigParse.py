def GetConfigFile(config_filename=None, directory=None):
    import os
    import sys

    if directory:
        script_path = directory
    else:
        script_path = os.path.abspath(os.path.dirname(sys.argv[0]))

    import glob

    if config_filename:
        # Relative filename
        config_file = glob.glob(
            script_path + "/{}".format(config_filename), recursive=False
        )

        # Absolute filename
        if len(config_file) == 0:
            config_file = glob.glob(config_filename)
    else:
        config_file = glob.glob(script_path + "/**/*.cfg", recursive=False)

    if len(config_file) == 0:
        config_file = glob.glob(script_path + "/*.cfg", recursive=False)

    if len(config_file) == 0:
        import TensorNAS

        tensornas_path = os.path.dirname(TensorNAS.__file__)
        config_file = glob.glob(
            tensornas_path + "/**/{}".format(config_filename), recursive=True
        )

    if len(config_file) == 0:
        raise Exception("Config file not found")

    print("Config location: {}".format(config_file))

    return config_file


def LoadConfig(config_file):
    import configparser

    config = configparser.ConfigParser()
    config.read(config_file)

    return config


def CopyConfig(config_filename, test_name):
    from shutil import copyfile
    from pathlib import Path

    path = "Output/{}".format(test_name)
    Path(path).mkdir(parents=True, exist_ok=True)
    copyfile(config_filename[-1], path + "/config.cfg")


def _GetFloat(parent, item):
    if not parent:
        return None

    if item not in parent:
        return 0.1

    return float(parent[item])


def _GetInt(parent, item, default=0):
    if not parent:
        return None

    if item not in parent:
        return default

    return int(parent[item])


def _GetBool(parent, item, default=False):
    if not parent:
        return default

    ret = parent.getboolean(item)

    if ret is None:
        return default

    return ret


def _GetStr(parent, item):
    if not parent:
        return None

    ret = str(parent[item])

    if ret != "None":
        return ret
    else:
        return None


def _GetGeneral(config):
    return config["general"]


def GetBlockArchitecture(config):
    return _GetGeneral(config)["BlockArchitecture"]


def GetClassCount(config):
    return int(_GetGeneral(config)["ClassCount"])


def GetVerbose(config):
    return _GetGeneral(config).getboolean("Verbose")


def GetMultithreaded(config):
    return _GetGeneral(config).getboolean("Multithreaded")


def GetDistributed(config):
    return _GetGeneral(config).getboolean("Distributed")


def GetDatasetModule(config):
    return _GetGeneral(config)["DatasetModule"]


def GetUseDatasetDirectory(config):
    try:
        return _GetStr(_GetGeneral(config), "DatasetDirectory")
    except Exception as e:
        return False


def GetDatasetDirectory(config):
    return _GetStr(_GetGeneral(config), "DatasetDirectory")


def GetLocalDataset(config):
    ret = _GetGeneral(config).getboolean("LocalDataset")
    if ret is None:
        ret = False
    return ret


def GetGenBlockArchitecture(config):
    return _GetGeneral(config)["GenBlockArchitecture"]


def GetThreadCount(config):
    return int(_GetGeneral(config)["ThreadCount"])


def GetGPU(config):
    return _GetGeneral(config).getboolean("GPU")


def GetLog(config):
    return _GetGeneral(config).getboolean("Log")


def _GetEvolution(config):
    return config["evolution"]


def GetCrossoverProbability(config):
    return float(_GetEvolution(config)["CrossoverProbability"])


def GetVerboseMutation(config):
    return _GetEvolution(config).getboolean("VerboseMutation")


def GetMutationMethod(config):
    return _GetStr(_GetEvolution(config), "MutationMethod")


def GetVariableMutationGenerationalChange(config):
    return float(_GetEvolution(config)["VariableMutationGenerationalChange"])


def GetMutationAttempts(config):
    return int(_GetEvolution(config)["MutationAttempts"])


def GetRetrainEveryGeneration(config):
    return _GetEvolution(config).getboolean("RetrainEveryGeneration")


def GetUseReinforcementLearning(config):
    return _GetBool(_GetEvolution(config), "UseReinforcementLearningMutation")


def GetAlpha(config):
    return float(_GetEvolution(config)["Alpha"])


def GetMutationProbability(config):
    try:
        return float(_GetEvolution(config)["MutationProbability"])
    except Exception:
        return 0.0


def GetSelfMutationProbability(config):
    try:
        return float(_GetEvolution(config)["SelfMutationProbability"])
    except Exception:
        return 0.2


def GetPopulationSize(config):
    return int(_GetEvolution(config)["PopulationSize"])


def GetGenerationCount(config):
    return int(_GetEvolution(config)["GenerationCount"])


def _GetOutput(config):
    return config["output"]


def GetGenerationGap(config):
    return int(_GetOutput(config)["GenerationGap"])


def GetGenerationSaveInterval(config):
    val = _GetOutput(config)["GenerationSave"]

    if val == "INTERVAL":
        return _GetGenerationSaveInterval(config)
    else:
        return 1


def _GetGenerationSaveInterval(config):
    return int(_GetOutput(config)["GenerationSaveInterval"])


def GetFigureTitle(config):
    return _GetOutput(config)["FigureTitle"]


def GetSaveIndividual(config):
    return _GetOutput(config).getboolean("SaveIndividuals")


def GetOutputPrefix(config):
    return _GetOutput(config)["OutputPrefix"]


def _GetGoals(config):
    return config["goals"]


def _GetFilters(config):
    return config["filter"]


def _GetGoalsNames(config):
    goals = _GetStr(_GetGoals(config), "GoalsNames").split(",")
    for i in range(len(goals)):
        goals[i] = goals[i].strip()
    return goals


def _GetOptimizationGoals(config):
    goal_names = _GetGoalsNames(config)
    weights = GetWeights(config)
    OptimizationGoal = dict()

    for count, (name, weight) in enumerate(zip(goal_names, weights)):
        name = name.upper()

        if weight == 1:
            name = name + "_DOWN"
        else:
            name = name + "_UP"

        OptimizationGoal[name] = count

    return OptimizationGoal


def _GetLogString(config):
    log_strings = []
    goals_names = _GetGoalsNames(config)

    mutation_log_string = ""
    evaluated_values_log_string = ""
    pareto_log_string = ""
    raw_evaluated_values_row = []

    for i in range(len(goals_names)):
        mutation_log_string = mutation_log_string + goals_names[i] + " diff: {} "
        evaluated_values_log_string = (
            evaluated_values_log_string + goals_names[i] + ":{}, "
        )
        pareto_log_string = (
            pareto_log_string + goals_names[len(goals_names) - 1 - i] + ": {}, "
        )
        raw_evaluated_values_row.append([goals_names[i]])

    return (
        mutation_log_string,
        evaluated_values_log_string,
        pareto_log_string,
        raw_evaluated_values_row,
    )


def _GetGoalVector(config):
    import ast

    return [n for n in ast.literal_eval(_GetGoals(config)["GoalVector"])]


def _GetNormalizationVector(config):
    import ast

    return [n for n in ast.literal_eval(_GetGoals(config)["NormalizationVector"])]


def GetFilterFunction(config):
    import importlib

    module = importlib.import_module(_GetFilterFunctionModule(config))
    func = getattr(module, _GetFilters(config)["FilterFunction"])

    return func


def GetUseGoalAttainment(config):
    return _GetBool(_GetFilters(config), "UseGoalAttainment", True)


def _GetFilterFunctionModule(config):
    return _GetFilters(config)["FilterFunctionModule"]


def GetGoalsNumber(config):
    return int(_GetGoals(config)["GoalsNumber"])


def GetWeights(config):
    config_arg = _GetFilters(config)["Weights"]

    w_len = GetGoalsNumber(config)

    if config_arg == "minimize":
        return [-1] * w_len
    elif config_arg == "maximize":
        return [1] * w_len
    else:
        import ast

        return [
            n.strip() if isinstance(n, str) else n for n in ast.literal_eval(config_arg)
        ]


def GetFilterFunctionArgs(config):
    # Goal vector varies, normilization vector is static
    goal_vector = _GetGoalVector(config)
    n_vector = _GetNormalizationVector(config)
    return goal_vector, n_vector


def _GetTensorflow(config):
    return config["tensorflow"]


def GetTrainingSampleSize(config):
    return _GetInt(_GetTensorflow(config), "TrainingSampleSize")


def GetTestSampleSize(config):
    return _GetInt(_GetTensorflow(config), "TestSampleSize")


def GetValidationSampleSize(config):
    return _GetInt(_GetTensorflow(config), "ValidationSampleSize")


def GetValidationSplit(config):
    return _GetFloat(_GetTensorflow(config), "ValidationSplit")


def GetTFOptimizer(config):
    return _GetTensorflow(config)["Optimizer"]


def GetTFLoss(config):
    return _GetTensorflow(config)["Loss"]


def GetTFMetrics(config):
    return _GetStr(_GetTensorflow(config), "Metrics").split()


def GetTFEarlyStopper(config):
    return _GetBool(_GetTensorflow(config), "EarlyStopper")


def GetTFPatience(config):
    return _GetInt(_GetTensorflow(config), "Patience")


def GetTFStopperMonitor(config):
    return _GetStr(_GetTensorflow(config), "StopperMonitor")


def GetTFStopperMinDelta(config):
    return _GetFloat(_GetTensorflow(config), "StopperMinDelta")


def GetTFStopperMode(config):
    return _GetStr(_GetTensorflow(config), "StopperMode")


def GetTFBatchSize(config):
    return _GetInt(_GetTensorflow(config), "BatchSize")


def GetTFTestBatchSize(config):
    return _GetInt(_GetTensorflow(config), "TestBatchSize")


def GetTFEpochs(config):
    return int(_GetTensorflow(config)["Epochs"])


def GetTFQuantizationAware(config):
    return _GetTensorflow(config).getboolean("QuantizationAware")


def GetTFUseClearMemory(config):
    return _GetBool(_GetTensorflow(config), "UseClearMemory")


def _GetLRScheduler(config):
    try:
        return config["lrscheduler"]
    except KeyError:
        return None


def GetUseLRScheduler(config):
    return _GetBool(_GetLRScheduler(config), "UseLRScheduler")


def GetLRScheduler(config):
    return _GetStr(_GetLRScheduler(config), "LRScheduler")


def GetLRInitialLearningRate(config):
    return _GetFloat(_GetLRScheduler(config), "InitialLearningRate")


def GetLRDecayPerEpoch(config):
    return _GetFloat(_GetLRScheduler(config), "DecayPerEpoch")


def _GetImageDataGeneartor(config):
    try:
        return config["image data generator"]
    except KeyError:
        return None


def UseImageDataGenerator(config):
    if _GetImageDataGeneartor(config):
        return True
    return False


def GetRotationRange(config):
    return _GetInt(_GetImageDataGeneartor(config), "RotationRange")


def GetWidthShiftRange(config):
    return _GetFloat(_GetImageDataGeneartor(config), "WidthShiftRange")


def GetHeightShiftRange(config):
    return _GetFloat(_GetImageDataGeneartor(config), "HeightShiftRange")


def GetHorizontalFlip(config):
    return _GetBool(_GetImageDataGeneartor(config), "HorizontalFlip")


def GetImageDataGeneratorValidationSplit(config):
    return _GetFloat(_GetImageDataGeneartor(config), "ValidationSplit")


def _GetMlonmcu(config):
    return config["mlonmcu"]

def GetUseMlonmcu(config):
    return _GetBool(_GetMlonmcu(config), "UseMLonMCU")

def GetMlonmcuMetrics(config):
    metrics = _GetStr(_GetMlonmcu(config), "metrics").split(",")
    for i in range(len(metrics)):
        metrics[i] = metrics[i].strip()
    return metrics


def GetMlonmcuPlatform(config):
    platforms = _GetStr(_GetMlonmcu(config), "platform").split(",")
    for i in range(len(platforms)):
        platforms[i] = platforms[i].strip()
    return platforms


def GetMlonmcuBackend(config):
    backends = _GetStr(_GetMlonmcu(config), "backend").split(",")
    for i in range(len(backends)):
        backends[i] = backends[i].strip()
    return backends


def GetMlonmcuTarget(config):
    targets = _GetStr(_GetMlonmcu(config), "target").split(",")
    for i in range(len(targets)):
        targets[i] = targets[i].strip()
    return targets


def GetMlonmcuFrontend(config):
    frontends = _GetStr(_GetMlonmcu(config), "frontend").split(",")
    for i in range(len(frontends)):
        frontends[i] = frontends[i].strip()
    return frontends


def GetMlonmcuPostprocess(config):
    pprocess = _GetStr(_GetMlonmcu(config), "postprocess")
    if pprocess is not None:
        pprocess = pprocess.split(",")
        for i in range(len(pprocess)):
            pprocess[i] = pprocess[i].strip()
    return pprocess


def GetMlonmcuFeature(config):
    features = _GetStr(_GetMlonmcu(config), "feature")
    if features is not None:
        features = features.split(",")
        for i in range(len(features)):
            features[i] = features[i].strip()
    return features


def GetMlonmcuConfigs(config):
    configs = _GetStr(_GetMlonmcu(config), "configs")
    if configs is not None:
        configs = configs.split(",")
        for i in range(len(configs)):
            configs[i] = configs[i].strip()
    return configs


def GetMlonmcuParallel(config):
    parallel = _GetInt(_GetMlonmcu(config), "parllel")
    if parallel == 8:
        return parallel
    else:
        return None


def GetMlonmcuProgress(config):
    return _GetBool(_GetMlonmcu(config), "progress")


def GetMlonmcuVerbose(config):
    return _GetBool(_GetMlonmcu(config), "verbose")


def GetMlonmcuArgs(config):
    mlonmcu_args = {
        "metrics": None,
        "platform": None,
        "backend": None,
        "target": None,
        "frontend": None,
        "postprocess": None,
        "feature": None,
        "configs": None,
        "parllel": False,
    }
    mlonmcu_args["use_mlonmcu"] = GetUseMlonmcu(config)
    mlonmcu_args["metrics"] = GetMlonmcuMetrics(config)
    mlonmcu_args["platform"] = GetMlonmcuPlatform(config)
    mlonmcu_args["backend"] = GetMlonmcuBackend(config)
    mlonmcu_args["target"] = GetMlonmcuTarget(config)
    mlonmcu_args["frontend"] = GetMlonmcuFrontend(config)
    mlonmcu_args["postprocess"] = GetMlonmcuPostprocess(config)
    mlonmcu_args["feature"] = GetMlonmcuFeature(config)
    mlonmcu_args["configs"] = GetMlonmcuConfigs(config)
    mlonmcu_args["parllel"] = GetMlonmcuParallel(config)
    mlonmcu_args["progress"] = GetMlonmcuProgress(config)
    mlonmcu_args["verbose"] = GetMlonmcuVerbose(config)
    return mlonmcu_args


"""
/__TO DO__/

- postprocess, feature, configs : check input form validity
- check supported platforms, backends and targets by mlonmcu
    mlonmcu_supported_platforms =[]
    mlonmcu_supported_backends =[]
    mlonmcu_supported_targets =[]
    ans = p.run(['mlonmcu', 'flow', '--list-targets'], capture_output=True)
    ans = ans.stdout.decode()
    --> check it only one time in configparse
"""
