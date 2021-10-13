dataset_name = "Cifar10"
dataset_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def get_dataset():

    from TensorNASDemos.Datasets import (
        make_dataset_dirs,
        tmp_dir,
        zip_dir,
        bar_progress,
    )
    import wget, os

    make_dataset_dirs(dataset_name)

    wget.download(
        dataset_url,
        out=os.path.join(zip_dir, "{}.tar.gz".format(dataset_name)),
        bar=bar_progress,
    )

    output_dir = os.path.join(tmp_dir, dataset_name)

    import tarfile

    tar = tarfile.open(zip_dir + "/{}.tar.gz".format(dataset_name), "r:gz")
    tar.extractall(path=output_dir)
    tar.close()

    import shutil

    out_files = os.listdir(output_dir)

    sub_out_files = os.listdir(os.path.join(output_dir, out_files[0]))

    for file in sub_out_files:
        shutil.move(os.path.join(output_dir, out_files[0], file), output_dir)

    shutil.rmtree(os.path.join(output_dir, out_files[0]))

    from TensorNASDemos.Datasets.Cifar10.train import load_cifar_10_data

    (
        train_data,
        train_filename,
        train_labels,
        test_data,
        test_filenames,
        test_labels,
        label_names,
    ) = load_cifar_10_data(output_dir)

    return (
        train_data,
        train_filename,
        train_labels,
        test_data,
        test_filenames,
        test_labels,
        label_names,
    )
