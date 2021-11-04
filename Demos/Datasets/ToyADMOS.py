import os

car_download_prefix = "https://zenodo.org/record/3351307/files/ToyCar.7z.00"
car_download_count = 8
conveyor_download_prefix = "https://zenodo.org/record/3351307/files/ToyConveyor.7z.00"
conveyor_download_count = 7
train_download_prefix = "https://zenodo.org/record/3351307/files/ToyTrain.7z.00"
train_download_count = 9
download_suffix = "?download=1"


def _create_subdirs(parent_dir):

    # train
    from TensorNASDemos.Datasets import tmp_dir

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


def _pull_dataset(data_set_name):

    from TensorNASDemos.Datasets import tmp_dir, zip_dir, make_dataset_dirs

    if os.path.isdir(tmp_dir) == False:
        make_dataset_dirs(data_set_name)
        _create_subdirs(data_set_name)

    zip_folder_path = zip_dir + "/{}".format(data_set_name)
    if os.path.isdir(zip_folder_path) == False:
        os.mkdir(zip_folder_path)

    import glob

    zip_files = glob.glob(zip_folder_path + "/*7z*")

    if len(zip_files) < globals()[data_set_name + "_download_count"]:
        import wget
        from TensorNASDemos.Datasets import bar_progress

        for i in range(1, globals()[data_set_name + "_download_count"] + 1):
            print("")
            print(
                "{} #{}/{}".format(
                    data_set_name, i, globals()[data_set_name + "_download_count"]
                )
            )
            wget.download(
                globals()[data_set_name + "_download_prefix"]
                + str(i)
                + download_suffix,
                out=zip_folder_path + "/{}.7z.00{}".format(data_set_name, i),
                bar=bar_progress,
            )

        zip_files = glob.glob(zip_folder_path + "*7z*")

    return zip_files


def _unzip_zips(zips):

    import py7zr, pathlib

    if len(zips):
        output_dir = str(pathlib.Path(zips[0]).parent.resolve()) + "/data"
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
    else:
        raise Exception("No zips to unzip")

    if len(os.listdir(output_dir)) != 5:
        print("Unzipping, this could take a while")

        zips.sort()

        file = zips[0][:-4]
        with open(file, mode="ab") as outfile:
            for zip in zips:
                with open(zip, "rb") as infile:
                    outfile.write(infile.read())

        with py7zr.SevenZipFile(file, "r") as archive:
            archive.extractall(path=output_dir)

        import shutil

        out_files = os.listdir(output_dir)

        sub_out_files = os.listdir(os.path.join(output_dir, out_files[0]))

        for file in sub_out_files:
            shutil.move(os.path.join(output_dir, out_files[0], file), output_dir)

        shutil.rmtree(os.path.join(output_dir, out_files[0]))

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

    cases = ["case1", "case2", "case3", "case4"]
    sound_types = ["AnomalousSound_IND", "NormalSound_IND", "NormalSound_CNT"]

    target_fs = 16000

    dataset_out = {}

    for case in cases:
        for sound_type in sound_types:
            print("{} {}".format(case, sound_type))
            wav_dir = os.path.join(data_dir, case, sound_type)

            dataset_out[sound_type] = _wave_read_all(wav_dir, target_fs)

    wav_dir = os.path.join(data_dir, "EnvironmentalNoise_CNT")
    dataset_out["EnvironmentalNoise_CNT"] = _wave_read_all(wav_dir, target_fs)

    return dataset_out


def _get_dataset(dataset_name):
    from TensorNASDemos.Datasets import make_dataset_dirs

    # check directories exist
    make_dataset_dirs(dataset_name)
    _create_subdirs(dataset_name)

    # get zip files locations (download if not yet downloaded)
    dataset_zips = _pull_dataset(dataset_name)

    # unzip and merge zips
    data_dir = _unzip_zips(dataset_zips)

    # read in WAV files from disk
    data_set = read_in_dataset(data_dir)

    return data_set


def get_train_dataset():

    return _get_dataset("train")


def get_car_dataset():

    return _get_dataset("car")


def get_conveyor_dataset():

    return _get_dataset("conveyor")


def GetData():

    from random import randrange

    switch = randrange(0, 3)

    if switch == 0:
        return get_train_dataset()
    elif switch == 1:
        return get_conveyor_dataset()
    elif switch == 2:
        return get_car_dataset()
