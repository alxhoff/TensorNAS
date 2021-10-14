dataset_name = "VisalWakeWords"
dataset_zip = "cifar-10-python.tar.gz"
dataset_url = "https://www.cs.toronto.edu/~kriz/{}".format(dataset_zip)


def GetData():

    from TensorNASDemos.Datasets import (
        make_dataset_dirs,
        tmp_dir,
        zip_dir,
        bar_progress,
    )
    import wget, os

    make_dataset_dirs(dataset_name)

    if not os.path.isfile(os.path.join(zip_dir, dataset_zip)):
        print("Downloading Cifar10 dataset tar")
        wget.download(dataset_url, out=zip_dir, bar=bar_progress)
    else:
        print("Cifar10 tar already exists, skipping download")

    output_dir = os.path.join(tmp_dir, dataset_name)

    import tarfile

    if not len(os.listdir(output_dir)):
        print("Extracting Cifar10 tar")
        tar = tarfile.open(zip_dir + "/{}".format(dataset_zip), "r:gz")
        tar.extractall(path=output_dir)
        tar.close()

        import shutil

        out_files = os.listdir(output_dir)

        sub_out_files = os.listdir(os.path.join(output_dir, out_files[0]))

        for file in sub_out_files:
            shutil.move(os.path.join(output_dir, out_files[0], file), output_dir)

        shutil.rmtree(os.path.join(output_dir, out_files[0]))
    else:
        print("Cifar10 tar already extracted")

    from TensorNASDemos.Datasets.Cifar10.train import load_cifar_10_data

    (
        train_data,
        train_filenames,
        train_labels,
        test_data,
        test_filenames,
        test_labels,
        label_names,
    ) = load_cifar_10_data(output_dir)

    return (test_data, train_data, test_labels, train_labels, test_data[0].shape)
