import os

import numpy as np

ToyCar_download_prefix = "https://zenodo.org/record/3678171/files/"
ToyCar_zipfile = "dev_data_ToyCar.zip"
ToyCar_download_count = 1
download_suffix = "?download=1"


def _create_subdirs(parent_dir):

    # train
    from TensorNAS.Demos.Datasets import tmp_dir

    parent_dir = os.path.join(tmp_dir, parent_dir)
    train_dir = parent_dir + "/train_normal"
    if os.path.isdir(train_dir) == False:
        os.mkdir(train_dir)
    # test normal
    test_normal_dir = parent_dir + "/test_normal"
    if os.path.isdir(test_normal_dir) == False:
        os.mkdir(test_normal_dir)

    # test anomaly
    test_anomaly_dir = parent_dir + "/test_anomaly"
    if os.path.isdir(test_anomaly_dir) == False:
        os.mkdir(test_anomaly_dir)


def _pull_dataset(dataset_name):

    from TensorNAS.Demos.Datasets import tmp_dir, zip_dir, make_dataset_dirs

    if os.path.isdir(tmp_dir) == False:
        make_dataset_dirs(dataset_name)

    import glob

    zip_files = glob.glob(zip_dir + "/*{}*zip".format(dataset_name))

    if len(zip_files) < globals()[dataset_name + "_download_count"]:
        import wget
        from TensorNAS.Demos.Datasets import bar_progress

        print("")

        wget.download(
            globals()[dataset_name + "_download_prefix"]
            + globals()[dataset_name + "_zipfile"]
            + download_suffix,
            out=zip_dir,
            bar=bar_progress,
        )

        zip_files = glob.glob(zip_dir + "/*{}*zip".format(dataset_name))

    return zip_files


def _unzip_zips(zips, dataset_name):
    from TensorNAS.Demos.Datasets import tmp_dir

    if len(zips):
        output_dir = "{}/{}".format(tmp_dir, dataset_name)
    else:
        raise Exception("No zips to unzip")

    if len(os.listdir(output_dir)) != 2:
        import zipfile
        from tqdm import tqdm

        print("Unzipping, this could take a while")

        with zipfile.ZipFile(zips[0], "r") as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc="Extracting "):
                try:
                    zip_ref.extract(member, tmp_dir)
                except zipfile.error as e:
                    pass

    return output_dir


def _wavread(fn):
    from scipy.io import wavfile
    import numpy as np

    fs, data = wavfile.read(fn)
    data = data.astype(np.float32) / 2 ** (15)
    return data, fs


def _wave_read_all(wav_dir, target_fs):
    import glob
    from tqdm import tqdm
    import librosa

    ch_num = ["ch1", "ch2", "ch3", "ch4"]

    wav_file_set = []
    for jj in range(len(ch_num)):
        wav_dir_i = wav_dir + "/*" + ch_num[jj]
        wav_files = glob.glob(wav_dir_i + "*.wav")
        wav_file_set.append(wav_files)

    Num_wav = len(wav_files)
    S_all = []
    fn_all = []
    print("Loading...")
    for ii in tqdm(range(Num_wav)):
        signals = []
        for jj in range(len(ch_num)):
            fn = wav_file_set[jj][ii]
            signal, org_fs = _wavread(fn)
            signals.append(signal)
        signal *= 0
        for jj in range(len(ch_num)):
            signal += signals[jj]
        if org_fs != target_fs:
            signal = librosa.core.resample(signal, org_fs, target_fs)
        fn = fn[len(wav_dir) :].replace(ch_num[-1], "chAll")
        S_all.append(signal)
        fn_all.append(fn)
    return S_all, fn_all


def read_in_dataset(data_dir):
    from TensorNAS.Demos.Datasets.ToyADMOS.train import (
        file_list_generator,
        list_to_vector_array,
    )
    from TensorNAS.Demos.Datasets.ToyADMOS.common import yaml_load

    param = yaml_load()

    files = file_list_generator(data_dir)
    train_data = list_to_vector_array(
        files,
        msg="generate train_dataset",
        n_mels=param["feature"]["n_mels"],
        frames=param["feature"]["frames"],
        n_fft=param["feature"]["n_fft"],
        hop_length=param["feature"]["hop_length"],
        power=param["feature"]["power"],
    )

    return train_data


def GetData():
    from TensorNAS.Demos.Datasets import make_dataset_dirs

    dataset_name = "ToyCar"

    # check directories exist
    make_dataset_dirs(dataset_name)

    # get zip files locations (download if not yet downloaded)
    dataset_zips = _pull_dataset(dataset_name)

    # unzip and merge zips
    data_dir = _unzip_zips(dataset_zips, dataset_name)

    # read in WAV files from disk
    data_set = read_in_dataset(data_dir)

    return data_set, data_set[0].shape


def GetTestData():
    from TensorNAS.Demos.Datasets.ToyADMOS.common import (
        test_file_list_generator,
        get_machine_id_list_for_test,
        file_to_vector_array,
        yaml_load,
    )
    from TensorNAS.Demos.Datasets import tmp_dir
    import os

    dataset_dir = os.path.join(tmp_dir, "ToyCar")
    machine_ids = get_machine_id_list_for_test(dataset_dir)

    param = yaml_load()

    test_files = []
    y_true = []

    for id in machine_ids:
        tf, yt = test_file_list_generator(dataset_dir, id, True)
        test_files.append(tf)
        y_true.append(yt)

    test_files = np.concatenate(test_files)
    y_true = np.concatenate(y_true)

    test_data = []
    for tf in test_files:
        test_data.append(
            file_to_vector_array(
                tf,
                n_mels=param["feature"]["n_mels"],
                frames=param["feature"]["frames"],
                n_fft=param["feature"]["n_fft"],
                hop_length=param["feature"]["hop_length"],
                power=param["feature"]["power"],
            )
        )

    return test_data, y_true
