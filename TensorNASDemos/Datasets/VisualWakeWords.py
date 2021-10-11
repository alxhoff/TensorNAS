dataset_name = "VisalWakeWords"
dataset_zip = "vw_coco2014_96.tar.gz"
dataset_url = "https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/{}".format(
    dataset_zip
)


def get_dataset():

    from TensorNASDemos.Datasets import (
        make_dataset_dirs,
        tmp_dir,
        zip_dir,
        bar_progress,
    )
    import wget, os

    make_dataset_dirs(dataset_name)

    wget.download(dataset_url, out=zip_dir, bar=bar_progress)

    output_dir = os.path.join(tmp_dir, dataset_name)

    import tarfile

    tar = tarfile.open(zip_dir + "/{}".format(dataset_zip), "r:gz")
    tar.extractall(path=output_dir)
    tar.close()

    import shutil

    out_files = os.listdir(output_dir)

    sub_out_files = os.listdir(os.path.join(output_dir, out_files[0]))

    for file in sub_out_files:
        shutil.move(os.path.join(output_dir, out_files[0], file), output_dir)

    shutil.rmtree(os.path.join(output_dir, out_files[0]))

    from PIL import Image
    from numpy import asarray

    output_dataset = {}
    output_dataset["person"] = []
    output_dataset["non_person"] = []

    # person images
    person_dir = os.path.join(output_dir, "person")
    peron_images = os.listdir(person_dir)

    for image in peron_images:
        output_dataset["person"].append(
            asarray(Image.open(os.path.join(person_dir, image)))
        )

    # non-person images
    nonperson_dir = os.path.join(output_dir, "non_person")
    nonperson_images = os.listdir(nonperson_dir)

    for image in nonperson_images:
        output_dataset["non_person"].append(
            asarray(Image.open(os.path.join(nonperson_dir, image)))
        )

    return output_dataset


data = get_dataset()
