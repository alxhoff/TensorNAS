def ExportBlockArchitectureToJSON(ba, path):

    import json

    with open("{}/ba.json".format(path), "w+") as f:
        ba_dict = ba.toJSON()
        json.dump(ba_dict, f)


def ImportBlockArchitectureFromJSON(ba_json_loc):
    import json

    with open(ba_json_loc, "r") as f:
        try:
            ba = json.load(f)
        except Exception as e:
            print(e)
            raise e

        from TensorNAS.Core.Block import get_block_from_JSON

        try:
            ba = get_block_from_JSON(ba)
        except Exception as e:
            print(e)
            raise e

        return ba
    return None


def ImportGeneration(gen_folder):

    from os import listdir

    ind_folders = listdir(gen_folder)

    models = []

    for folder in ind_folders:
        models.append(
            ImportBlockArchitectureFromJSON("{}/{}/ba.json".format(gen_folder, folder))
        )

    return models


def GetBlockMod(blk_name):

    import TensorNAS, os, glob

    framework_path = os.path.dirname(os.path.dirname(TensorNAS.__file__))
    mod_name = glob.glob(
        "{}/**/{}.py".format(framework_path, blk_name),
        recursive=True,
    )[0][len(framework_path + "/") : -3].replace("/", ".")

    import importlib

    mod = importlib.import_module(mod_name)

    return mod
