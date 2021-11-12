dataset_name = "VisualWakeWords"
dataset_train_zip = "train2014.zip"
dataset_val_zip = "val2014.zip"
dataset_url = "http://images.cocodataset.org/zips/"
dataset_annotations_zip = "annotations_trainval2014.zip"
dataset_annotations_url = "http://images.cocodataset.org/annotations/"
TARGET_SIZE = 96


class DatasetImageWrapper:
    def __init__(self, dataset):

        self.dataset = dataset

    def __getitem__(self, item):
        import numpy as np

        return np.array(self.dataset[item][0])


class DatasetLabelWrapper:
    def __init__(self, dataset):

        self.dataset = dataset

    def __getitem__(self, item):

        return self.dataset[item][1]


def GetData():
    global TARGET_SIZE
    from TensorNAS.Demos import get_global

    batch_size = get_global("batch_size")

    from TensorNAS.Demos.Datasets import (
        make_dataset_dirs,
        tmp_dir,
        zip_dir,
        bar_progress,
    )
    import wget, os, zipfile, shutil

    make_dataset_dirs(dataset_name)

    if not os.path.isfile(os.path.join(zip_dir, dataset_train_zip)):
        print("Downloading VisualWakeWord training dataset zip")
        wget.download(dataset_url + dataset_train_zip, out=zip_dir, bar=bar_progress)
    else:
        print("VisualWakeWord training zip already exists, skipping download")

    if not os.path.isfile(os.path.join(zip_dir, dataset_val_zip)):
        print("Downloading VisualWakeWord validation dataset zip")
        wget.download(dataset_url + dataset_val_zip, out=zip_dir, bar=bar_progress)
    else:
        print("VisualWakeWord validation zip already exists, skipping download")

    output_dir = os.path.join(tmp_dir, dataset_name)

    if not len(os.listdir(output_dir)):
        for z in [dataset_train_zip, dataset_val_zip]:
            print("Extracting VisualWakeWord zip")

            with zipfile.ZipFile(zip_dir + "/{}".format(z), "r") as zip_ref:
                zip_ref.extractall(path=output_dir)

            out_subdirs = [f.path for f in os.scandir(output_dir) if os.path.isdir(f)]

            sub_out_files = os.listdir(os.path.join(output_dir, out_subdirs[0]))

            for file in sub_out_files:
                shutil.move(os.path.join(output_dir, out_subdirs[0], file), output_dir)

            shutil.rmtree(out_subdirs[0])
    else:
        print("VisualWakeWord zip already extracted")

    if not os.path.isfile(os.path.join(zip_dir, dataset_annotations_zip)):
        print("Downloading VisualWakeWord annotations")
        wget.download(
            dataset_annotations_url + dataset_annotations_zip,
            out=zip_dir,
            bar=bar_progress,
        )
    else:
        print("VisualWakeWord annotations zip already exists, skipping downloads")

    if not os.path.isdir(os.path.join(output_dir, "annotations")):
        import tqdm

        print("Extracting VisualWakeWord's annotation's zip")
        with zipfile.ZipFile(
            zip_dir + "/{}".format(dataset_annotations_zip), "r"
        ) as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc="Extracting "):
                try:
                    zip_ref.extract(member, output_dir)
                except zipfile.error as e:
                    pass
    else:
        print("VisualWakeWord's annotation's zip already extracted")

    train_annotations_file = os.path.realpath(
        os.path.expanduser(
            os.path.join(output_dir, "annotations/instances_train2014.json")
        )
    )
    val_annotations_file = os.path.realpath(
        os.path.expanduser(
            os.path.join(output_dir, "annotations/instances_val2014.json")
        )
    )
    annotations_output_dir = os.path.join(output_dir, "annotations_out")

    from TensorNAS.Demos.Datasets.VWW.create_coco_train_minival_split import (
        create_maxitrain_minival,
    )

    if not os.path.isdir(annotations_output_dir):
        os.mkdir(annotations_output_dir)

    if not os.path.isfile(
        os.path.join(annotations_output_dir, "instances_maxitrain.json")
    ) or not os.path.isfile(
        os.path.join(annotations_output_dir, "instances_minival.json")
    ):
        create_maxitrain_minival(
            train_annotations_file, val_annotations_file, annotations_output_dir
        )

    from TensorNAS.Demos.Datasets.VWW.create_visualwakewords_annotations import (
        create_visual_wakeword_annotations,
    )

    maxitrain_annotations_file = os.path.realpath(
        os.path.expanduser(
            os.path.join(annotations_output_dir, "instances_maxitrain.json")
        )
    )
    minival_annotations_file = os.path.realpath(
        os.path.expanduser(
            os.path.join(annotations_output_dir, "instances_minival.json")
        )
    )
    small_object_area_threshold = 0.005
    foreground_class_of_interest = "person"

    if not os.path.isfile(os.path.join(annotations_output_dir, "instances_train.json")):
        create_visual_wakeword_annotations(
            annotations_file=maxitrain_annotations_file,
            visualwakewords_annotations_path=os.path.join(
                annotations_output_dir, "instances_train.json"
            ),
            object_area_threshold=small_object_area_threshold,
            foreground_class_name=foreground_class_of_interest,
        )
        print(
            "Train annotations for {} class created".format(
                foreground_class_of_interest
            )
        )
    else:
        print("Train annotations already exist")

    if not os.path.isfile(os.path.join(annotations_output_dir, "instances_val.json")):
        create_visual_wakeword_annotations(
            annotations_file=minival_annotations_file,
            visualwakewords_annotations_path=os.path.join(
                annotations_output_dir, "instances_val.json"
            ),
            object_area_threshold=small_object_area_threshold,
            foreground_class_name=foreground_class_of_interest,
        )
        print(
            "Validation annotations for {} class created".format(
                foreground_class_of_interest
            )
        )
    else:
        print("Val annotations already exist")

    import pyvww

    train_dataset = pyvww.pytorch.VisualWakeWordsClassification(
        root=output_dir,
        annFile=os.path.join(annotations_output_dir, "instances_train.json"),
    )
    test_dataset = pyvww.pytorch.VisualWakeWordsClassification(
        root=output_dir,
        annFile=os.path.join(annotations_output_dir, "instances_val.json"),
    )

    print("Resizing images and saving datasets to disk")

    # save dataset into person and non-person directories
    for cur_dataset in ["train_dataset", "test_dataset"]:
        data_out_dir = os.path.join(output_dir, cur_dataset)
        if not os.path.isdir(data_out_dir):
            os.mkdir(data_out_dir)

        if not os.path.isdir(os.path.join(data_out_dir, "person")):
            os.mkdir(os.path.join(data_out_dir, "person"))

        if not os.path.isdir(os.path.join(data_out_dir, "nonperson")):
            os.mkdir(os.path.join(data_out_dir, "nonperson"))

        nonperson_dir = os.path.join(data_out_dir, "nonperson")
        person_dir = os.path.join(data_out_dir, "person")

        cur_dataset_len = len(eval("{}".format(cur_dataset)))

        import sys

        cur_image_count = len(os.listdir(person_dir)) + len(os.listdir(nonperson_dir))

        if not (cur_dataset_len == cur_image_count):
            for i, img in enumerate(eval("{}".format(cur_dataset))):
                sys.stdout.write(
                    "Processing %s image: %d/%d   \r"
                    % (cur_dataset, i, cur_dataset_len)
                )
                # nonperson
                if not img[1]:
                    if not os.path.isfile(
                        os.path.join(nonperson_dir, "{}.jpg".format(i))
                    ):
                        resize_save_image(img[0], i, nonperson_dir)
                else:
                    if not os.path.isfile(os.path.join(person_dir, "{}.jpg".format(i))):
                        resize_save_image(img[0], i, person_dir)

    print("Images resized and written to disk")

    # create keras dataset from directory
    import tensorflow as tf

    BATCH_SIZE = batch_size
    validation_split = 0.1

    train_dir = os.path.join(output_dir, "train_dataset")
    val_dir = os.path.join(output_dir, "test_dataset")

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=validation_split,
        rescale=1.0 / 255,
    )
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=BATCH_SIZE,
        subset="training",
        color_mode="rgb",
    )
    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=BATCH_SIZE,
        subset="validation",
        color_mode="rgb",
    )
    print(train_generator.class_indices)

    return train_generator, val_generator, train_generator.image_shape


def resize_save_image(image, save_index, out_dir):
    global TARGET_SIZE
    from PIL import Image

    wpercent = TARGET_SIZE / float(image.size[0])
    hsize = int((float(image.size[1]) * float(wpercent)))
    image = image.resize((TARGET_SIZE, hsize), Image.ANTIALIAS)
    image.save("{}/{}.jpg".format(out_dir, save_index))

    pass
