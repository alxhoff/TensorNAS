def GetOptimizer(optimizer_name=None):

    if optimizer_name is None:
        return None

    import os, TensorNAS, glob

    framework_dir = os.path.dirname(os.path.dirname(TensorNAS.__file__))
    framework_path = os.path.abspath(os.path.dirname(TensorNAS.__file__))
    gb = glob.glob(
        "{}/**/{}.py".format(
            framework_path, optimizer_name[0].lower() + optimizer_name[1:]
        ),
        recursive=True,
    )
    mod_name = gb[0][len(framework_dir + "/") : -3].replace("/", ".")

    import importlib

    opt_mod = importlib.import_module(mod_name)

    opt = opt_mod.Optimizer()

    return opt
