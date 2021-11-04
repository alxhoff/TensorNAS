import math
import random
from enum import Enum, auto

from TensorNAS.Tools.Util import _generate_permutations, dimension_mag


class MutationOperators(Enum):
    STEP = auto()
    RANDOM = auto()

    SYNC_STEP = auto()  # Both values in the tuple are mutated together
    SYNC_RANDOM = auto()


def mutate_int_square(val, min_bound, max_bound, operator=MutationOperators.STEP):
    if operator == MutationOperators.STEP:
        if random.randrange(0, 2):  # multiplay
            if val <= max_bound / 2:
                return val * 2
        else:  # divide
            if val > 1:
                return val / 2
    elif operator == MutationOperators.RANDOM:
        max_exponent = math.floor(math.log(max_bound) / math.log(2))
        exponent = random.randrange(1, max_exponent + 1)
        return math.pow(2, exponent)


def mutate_float(
    val, min_bound, max_bound, operator=MutationOperators.STEP, step_size=0.001
):

    if operator == MutationOperators.RANDOM:
        return round(random.uniform(min_bound, max_bound))
    elif operator == MutationOperators.STEP:
        if random.randrange(0, 2) and val < max_bound:
            return val + step_size
        else:
            if val > min_bound:
                return val - step_size
            else:
                return val + step_size


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


def mutate_enum_i(enum):
    return random.choice(list(enum))
