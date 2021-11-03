from TensorNASDemos.Datasets import tmp_dir
import os

data_folder = "SpeechCommands"
data_dir = os.path.join(tmp_dir, data_folder)


def GetData():

    from TensorNASDemos.Datasets.SpeechCommands.kws_util import parse_command
    from TensorNASDemos.Datasets.SpeechCommands.get_dataset import get_training_data
    import tensorflow as tf

    Flags, unparsed = parse_command()
    Flags.data_dir = data_dir

    ds_train, ds_test, ds_val = get_training_data(Flags)

    train_len = len(ds_train)
    val_len = len(ds_val)
    train_shuffle_buffer_size = 85511
    val_shuffle_buffer_size = 10102
    # test_shuffle_buffer_size = 4890

    ds_train = ds_train.take(train_shuffle_buffer_size).repeat()
    ds_val = ds_val.take(val_shuffle_buffer_size).repeat()

    import tensorflow as tf

    shape = tuple(tf.compat.v1.data.get_output_shapes(ds_train)[0].as_list()[1:])

    return ds_train, ds_val, shape, train_len, val_len
