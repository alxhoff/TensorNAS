def GetOptimizer(optimizer_name=None):

    if optimizer_name is None:
        return None

    import os, TensorNAS, glob

    framework_path = os.path.dirname(os.path.dirname(TensorNAS.__file__))
    mod_name = glob.glob(
        "{}/**/{}.py".format(
            framework_path, optimizer_name[0].lower() + optimizer_name[1:]
        ),
        recursive=True,
    )[0][len(framework_path + "/") : -3].replace("/", ".")

    import importlib

    opt_mod = importlib.import_module(mod_name)

    opt = opt_mod.Optimizer()

    return opt
