from TensorNASDemos.Datasets import tmp_dir
import os

data_folder = "SpeechCommands"
data_dir = os.path.join(tmp_dir, data_folder)


def GetData():

    from TensorNASDemos.Datasets.SpeechCommands.kws_util import parse_command
    from TensorNASDemos.Datasets.SpeechCommands.get_dataset import get_training_data

    Flags, unparsed = parse_command()
    Flags.data_dir = data_dir

    ds_train, ds_test, ds_val = get_training_data(Flags)

    train_shuffle_buffer_size = 85511
    val_shuffle_buffer_size = 10102
    test_shuffle_buffer_size = 4890

    ds_train = ds_train.shuffle(train_shuffle_buffer_size)
    ds_val = ds_val.shuffle(val_shuffle_buffer_size)
    ds_test = ds_test.shuffle(test_shuffle_buffer_size)

    return ds_train, ds_test, ds_val, None
