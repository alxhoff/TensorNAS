from TensorNASDemos.Datasets import tmp_dir
import os

data_folder = "SpeechCommands"
data_dir = os.path.join(tmp_dir, data_folder)


def get_dataset():

    from KWS.kws_util import parse_command
    from KWS.get_dataset import get_training_data

    Flags, unparsed = parse_command()
    Flags.data_dir = data_dir

    ds_train, ds_test, ds_val = get_training_data(Flags)

    return ds_train, ds_test, ds_val
