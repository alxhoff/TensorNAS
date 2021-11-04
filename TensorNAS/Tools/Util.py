import math
import random
from functools import reduce
from pkgutil import iter_modules


def dimension_mag(dimension):
    return reduce((lambda x, y: x * y), dimension)


def _find_prime_factors(product):
    primeFactors = []
    while not product % 2:
        primeFactors.append(2)
        product //= 2
    for i in range(3, int(math.sqrt(product))):
        while not product % i:
            primeFactors.append(i)
            product //= i
    if product > 2:
        primeFactors.append(product)
    return primeFactors


def _generate_permutations(product, item_count):
    prime_factors = _find_prime_factors(product)
    while len(prime_factors) > item_count:
        index = random.randrange(0, len(prime_factors) - 1)
        prime_factors[index : index + 2] = [
            prime_factors[index] * prime_factors[index + 1]
        ]
    return prime_factors


def find_modules(pkg, dir):
    from importlib import import_module

    modules = []
    for mod in iter_modules([dir]):
        if not mod.ispkg:
            mod_name = pkg + "." + mod.name
            modules.append(import_module(mod_name))
    return modules


def custom_sparse_categorical_accuracy(y_true, y_pred):
    from tensorflow.keras import backend as K

    return K.cast(
        K.equal(K.max(y_true, axis=-1), K.cast(K.argmax(y_pred, axis=-1), K.floatx())),
        K.floatx(),
    )


def block_width(block):
    try:
        return block.index("\n")
    except ValueError:
        return len(block)


def stack_str_blocks(blocks):
    import itertools

    builder = []
    block_lens = [block_width(bl) for bl in blocks]
    split_blocks = [bl.split("\n") for bl in blocks]

    for line_list in itertools.zip_longest(*split_blocks, fillvalue=None):
        for i, line in enumerate(line_list):
            if line is None:
                builder.append(" " * block_lens[i])
            else:
                builder.append(line)
            if i != len(line_list) - 1:
                builder.append(" ")  # Padding
        builder.append("\n")

    return "".join(builder[:-1])


def list_available_blocks():
    from TensorNAS.BlockTemplates import SupportedBlocks

    for block in SupportedBlocks:
        print(block.value)


def list_available_block_architectures():
    from TensorNAS.BlockTemplates import SupportedArchitectureBlocks

    for arch in SupportedArchitectureBlocks:
        print(arch.value)


def save_block_architecture(ba, test_name, model_name, logger):
    from pathlib import Path
    import os
    from TensorNAS.Tools.JSONImportExport import ExportBlockArchitectureToJSON

    if logger:
        logger.log("Saving new model, name:{}".format(model_name))

    path = "Output/{}/Models/{}".format(test_name, model_name)
    if not os.path.isdir(path):
        Path(path).mkdir(parents=True, exist_ok=True)
    ExportBlockArchitectureToJSON(ba, path)


def save_model(model, test_name, model_name, logger):
    from pathlib import Path
    import os

    if logger:
        logger.log("Saving new model, name:{}".format(model_name))

    path = "Output/{}/Models/{}".format(test_name, model_name)
    if not os.path.isdir(path):
        Path(path).mkdir(parents=True, exist_ok=True)
    model.save(path)

    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    open(
        "Output/{}/Models/{}/saved_model.tflite".format(test_name, model_name), "wb"
    ).write(tflite_model)


def copy_output_model(test_name, gen, index_from, index_to):

    from_subdir = "Models/{}/{}".format(gen - 1, index_from)
    to_subdir = "Models/{}/{}".format(gen, index_to)

    copy_model(test_name, from_subdir, to_subdir)


def copy_pareto_model(test_name, gen, index_from, index_to):

    from_subdir = "Models/{}/{}".format(gen, index_from)
    to_subdir = "Models/pareto/{}".format(index_to)

    copy_model(test_name, from_subdir, to_subdir)


def copy_model(test_name, from_subdir, to_subdir):

    from_path = "Output/{}/{}".format(test_name, from_subdir)
    to_path = "Output/{}/{}".format(test_name, to_subdir)

    import os, distutils.dir_util
    from pathlib import Path

    if os.path.isdir(from_path):
        if not os.path.isdir(to_path):
            Path(to_path).mkdir(parents=True, exist_ok=True)
        distutils.dir_util.copy_tree(from_path, to_path)


def shortcut(input, residual):
    import tensorflow as tf

    input_shape = tf.keras.backend.int_shape(input)
    residual_shape = tf.keras.backend.int_shape(residual)
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input

    if (input_shape[1] % residual_shape[1] or input_shape[2] % residual_shape[2]) and (
        input_shape[1] != residual_shape[1] and input_shape[2] != residual_shape[2]
    ):
        kernel_size = (
            input_shape[1] - residual_shape[1] + 1,
            input_shape[2] - residual_shape[2] + 1,
        )
        stride_width = 1
        stride_height = 1
    else:
        kernel_size = (1, 1)
        stride_width = int(round(input_shape[1] / residual_shape[1]))
        stride_height = int(round(input_shape[2] / residual_shape[2]))

    if not equal_channels:
        shortcut = tf.keras.layers.Conv2D(
            filters=residual_shape[3],
            kernel_size=kernel_size,
            strides=(stride_width, stride_height),
            padding="valid",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        )(input)

    ret = tf.keras.layers.add([shortcut, residual])
    return ret
