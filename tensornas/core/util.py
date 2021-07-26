import math
import random
from enum import Enum, auto
from functools import reduce
from pkgutil import iter_modules


class MutationOperators(Enum):
    STEP = auto()
    RANDOM = auto()

    SYNC_STEP = auto()  # Both values in the tuple are mutated together
    SYNC_RANDOM = auto()


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


def mutate_dimension(intput_dim):
    while True:
        new_dim = _generate_permutations(dimension_mag(intput_dim), len(intput_dim))
        if new_dim != intput_dim:
            return new_dim


def mutate_int(val, min_bound, max_bound, operator=MutationOperators.STEP):
    if operator == MutationOperators.RANDOM:
        return random.randrange(min_bound, max_bound + 1)
    elif operator == MutationOperators.STEP:
        if random.randrange(0, 2) and val < max_bound:
            return val + 1
        else:
            if val > min_bound:
                return val - 1
            else:
                return val + 1


def mutate_unit_interval(
    val, min_bound, max_bound, operator=MutationOperators.STEP, step_size=0.05
):
    if operator == MutationOperators.RANDOM:
        return random.random()
    elif operator == MutationOperators.STEP:
        if random.randrange(0, 2) and val <= max_bound - step_size:
            return val + step_size
        else:
            if val >= min_bound + step_size:
                return val - step_size
            else:
                return val + step_size


def mutate_tuple(val, min_bound, max_bound, operator=MutationOperators.SYNC_STEP):
    while True:  # loop until a mutation was performed
        if operator == MutationOperators.STEP:
            if random.randrange(0, 2):  # Inc
                if random.randrange(0, 2):  # X value
                    if val[0] < max_bound:
                        return val[0] + 1, val[1]
                    else:
                        continue
                else:  # Y value
                    if val[0] > 1:
                        return val[0], val[1] + 1
                    else:
                        continue
            else:  # Dec
                if random.randrange(0, 2):  # X value
                    if val[0] < max_bound:
                        return val[0] - 1, val[1]
                    else:
                        continue
                else:  # Y value
                    if val[0] > 1:
                        return val[0], val[1] - 1
                    else:
                        continue
        elif operator == MutationOperators.SYNC_STEP:
            if random.randrange(0, 2):  # Inc
                if val[0] < max_bound and val[1] < max_bound:
                    return val[0] + 1, val[1] + 1
                else:
                    continue
            else:  # Dec
                if val[0] > min_bound and val[1] > min_bound:
                    return val[0] - 1, val[1] - 1
                else:
                    continue
        elif operator == MutationOperators.SYNC_RANDOM:
            while (
                True
            ):  # generate a different value to what we currently have (referenced using X val)
                val = random.randrange(min_bound, max_bound + 1)
                if val != val[0]:
                    break
            return val, val
        elif operator == MutationOperators.RANDOM:
            if random.randrange(0, 2):  # X
                while True:
                    val = random.randrange(min_bound, max_bound + 1)
                    if val != val[0]:
                        break
                return val, val[1]
            else:  # Y
                while True:
                    val = random.randrange(min_bound, max_bound + 1)
                    if val != val[0]:
                        break
                return val[0], val


def mutate_enum(val, enum):
    while True:
        new_val = random.choice(list(enum))
        if new_val != val:
            return new_val


# enum becomes a datatype and is used to access the static constants whose value is known at compile type
def mutate_enum_i(enum):
    return random.choice(list(enum))


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
    from tensornas.blocktemplates import SupportedBlocks

    for block in SupportedBlocks:
        print(block.value)


def list_available_block_architectures():
    from tensornas.blocktemplates import SupportedArchitectureBlocks

    for arch in SupportedArchitectureBlocks:
        print(arch.value)


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

    from_path = "Output/{}/Models/{}/{}".format(test_name, gen - 1, index_from)
    to_path = "Output/{}/Models/{}/{}".format(test_name, gen, index_to)
    import os, distutils.dir_util
    from pathlib import Path

    if os.path.isdir(from_path):
        if not os.path.isdir(to_path):
            Path(to_path).mkdir(parents=True, exist_ok=True)
        # shutil.copytree(from_path, to_path)
        distutils.dir_util.copy_tree(from_path, to_path)
