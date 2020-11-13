import json
import os


def write_nas_model_json(json_model, filename):
    with open("{}.json".format(filename), mode="w+") as f:
        f.write(json.dumps(json_model, indent=2))


def load_nas_model_json(filename):
    if os.path.isfile("{}.json".format(filename)):
        with open("{}.json".format(filename)) as f:
            return json.load(f)
