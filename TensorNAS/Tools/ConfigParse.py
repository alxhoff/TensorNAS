def GetConfigFile(config_filename=None, directory=None):
    import os, sys

    if directory:
        script_path = directory
    else:
        script_path = os.path.abspath(os.path.dirname(sys.argv[0]))

    import glob

    if config_filename:
        config_file = glob.glob(
            script_path + "/{}.cfg".format(config_filename), recursive=False
        )
    else:
        config_file = glob.glob(script_path + "/**/*.cfg", recursive=False)

    if len(config_file) == 0:
        config_file = glob.glob(script_path + "/*.cfg", recursive=False)

    if len(config_file) == 0:
        import TensorNAS

        tensornas_path = os.path.dirname(TensorNAS.__file__)
        config_file = glob.glob(
            tensornas_path + "/**/{}.cfg".format(config_filename), recursive=True
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

    config_file = GetConfigFile(config_filename)

    from shutil import copyfile
    from pathlib import Path

    path = "Output/{}".format(test_name)
    Path(path).mkdir(parents=True, exist_ok=True)
    copyfile(config_file[-1], path + "/config.cfg")


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


def GetMutationProbability(config):

    return float(_GetEvolution(config)["MutationProbability"])


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

    return _GetOutput(config)["SaveIndividuals"]


def GetOutputPrefix(config):

    return _GetOutput(config)["OutputPrefix"]


def _GetGoals(config):

    return config["goals"]


def _GetFilters(config):

    return config["filter"]


def _GetVariableGoal(config):

    return _GetGoals(config).getboolean("VariableGoal")


def _GetNormalizationVectorStart(config):

    return int(_GetGoals(config)["NormalizationVectorStart"])


def _GetNormalizationVectorEnd(config):

    return int(_GetGoals(config)["NormalizationVectorEnd"])


def _GetNormalizationVectorSteps(config):

    return int(_GetGoals(config)["NormalizationVectorSteps"])


def _GetGoalVector(config):

    import ast

    return [n for n in ast.literal_eval(_GetGoals(config)["GoalVector"])]


def _GetGoalVectorStart(config):

    return int(_GetGoals(config)["GoalVectorStart"])


def _GetGoalVectorEnd(config):

    return int(_GetGoals(config)["GoalVectorEnd"])


def _GetGoalVectorSteps(config):

    return int(_GetGoals(config)["GoalVectorSteps"])


def _GetNormalizationVector(config):

    import ast

    return [n for n in ast.literal_eval(_GetGoals(config)["NormalizationVector"])]


def GetFilterFunction(config):
    import importlib

    module = importlib.import_module(_GetFilterFunctionModule(config))
    func = getattr(module, _GetFilters(config)["FilterFunction"])

    return func


def _GetFilterFunctionModule(config):

    return _GetFilters(config)["FilterFunctionModule"]


def GetWeights(config):

    config_arg = _GetFilters(config)["Weights"]

    if _GetVariableGoal(config):
        w_len = _GetGoalVectorSteps(config)
    else:
        w_len = _GetNormalizationVectorSteps(config)

    if config_arg == "minimize":
        return [-1] * w_len
    elif config_arg == "maximize":
        return [1] * w_len
    else:
        import ast

        return [n.strip() for n in ast.literal_eval(config_arg)]


def _GenVectorsVaribleGoal(g_start, g_stop, g_step, n1, n2):

    if g_start == g_stop:
        goal_vectors = [(g_start, 100)]
    else:
        goal_vectors = [
            (i, 100)
            for i in range(g_start, g_stop + g_step, 1 if not g_step else g_step)
        ]
    normalization_vectors = [(n1, n2) for _ in range(len(goal_vectors))]

    return goal_vectors, normalization_vectors


def _GenVectorsVariableNormilization(n_start, n_stop, n_step, g1, g2):

    if n_start == n_stop:
        normalization_vectors = [(n_start, 1)]
    else:
        normalization_vectors = [
            (i, 1) for i in range(n_start, n_stop + n_step, 1 if not n_step else n_step)
        ]
    goal_vectors = [(g1, g2) for _ in range(len(normalization_vectors))]

    return goal_vectors, normalization_vectors


def GetFilterFunctionArgs(config):

    if _GetVariableGoal(config):
        # Goal vector varies, normilization vector is static
        g_start = _GetGoalVectorStart(config)
        g_stop = _GetGoalVectorEnd(config)
        g_step = int((g_stop - g_start) / (_GetGoalVectorSteps(config) - 1))
        n1, n2 = _GetNormalizationVector(config)
        return _GenVectorsVaribleGoal(g_start, g_stop, g_step, n1, n2)
    else:
        n_start = _GetNormalizationVectorStart(config)
        n_stop = _GetNormalizationVectorEnd(config)
        n_step = int((n_stop - n_start) / (_GetNormalizationVectorSteps(config) - 1))
        g1, g2 = _GetGoalVector(config)
        return _GenVectorsVariableNormilization(n_start, n_stop, n_step, g1, g2)


def _GetTensorflow(config):

    return config["tensorflow"]


def GetTrainingSampleSize(config):

    return int(_GetTensorflow(config)["TrainingSampleSize"])


def GetTestSampleSize(config):

    return int(_GetTensorflow(config)["TestSampleSize"])


def GetTFOptimizer(config):

    return _GetTensorflow(config)["Optimizer"]


def GetTFLoss(config):

    return _GetTensorflow(config)["Loss"]


def GetTFMetrics(config):

    return [_GetTensorflow(config)["Metrics"]]


def GetTFBatchSize(config):

    return int(_GetTensorflow(config)["BatchSize"])


def GetTFEpochs(config):
    return int(_GetTensorflow(config)["Epochs"])


def GetTFQuantizationAware(config):
    return _GetTensorflow(config).getboolean("QuantizationAware")
