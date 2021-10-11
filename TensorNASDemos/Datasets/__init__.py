import pathlib, os

tmp_dir = str(pathlib.Path(__file__).parent.resolve()) + "/tmp"
zip_dir = tmp_dir + "/zips"


def _make_tmp_dir():

    if os.path.isdir(tmp_dir) == False:
        os.mkdir(tmp_dir)

    if os.path.isdir(zip_dir) == False:
        os.mkdir(zip_dir)


def make_dataset_dirs(dataset_name):

    _make_tmp_dir()

    dir = tmp_dir + "/{}".format(dataset_name)

    if os.path.isdir(dir) == False:
        os.mkdir(dir)


def bar_progress(current, total, width=80):
    import sys

    progress_message = "Downloading: %d%% [%d / %d] bytes" % (
        current / total * 100,
        current,
        total,
    )
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()
