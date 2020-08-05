from enum import Enum, auto
import random


class MutationOperators(Enum):
    STEP = auto()
    RANDOM = auto()

    SYNC_STEP = auto()  # Both values in the tuple are mutated together
    SYNC_RANDOM = auto()


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


def mutate_tuple(tuple, min_bound, max_bound, operator=MutationOperators.SYNC_STEP):
    while True:  # loop until a mutation was performed
        if operator == MutationOperators.STEP:
            if random.randrange(0, 2):  # Inc
                if random.randrange(0, 2):  # X value
                    if tuple[0] < max_bound:
                        return (tuple[0] + 1, tuple[1])
                    else:
                        continue
                else:  # Y value
                    if tuple[0] > 1:
                        return (tuple[0], tuple[1] + 1)
                    else:
                        continue
            else:  # Dec
                if random.randrange(0, 2):  # X value
                    if tuple[0] < max_bound:
                        return (tuple[0] - 1, tuple[1])
                    else:
                        continue
                else:  # Y value
                    if tuple[0] > 1:
                        return (tuple[0], tuple[1] - 1)
                    else:
                        continue
        elif operator == MutationOperators.SYNC_STEP:
            if random.randrange(0, 2):  # Inc
                if tuple[0] < max_bound and tuple[1] < max_bound:
                    return (tuple[0] + 1, tuple[1] + 1)
                else:
                    continue
            else:  # Dec
                if tuple[0] > min_bound and tuple[1] > min_bound:
                    return (tuple[0] - 1, tuple[1] - 1)
                else:
                    continue
        elif operator == MutationOperators.SYNC_RANDOM:
            while (
                True
            ):  # generate a different value to what we currently have (referenced using X val)
                val = random.randrange(min_bound, max_bound + 1)
                if val != tuple[0]:
                    break
            return (val, val)
        elif operator == MutationOperators.RANDOM:
            if random.randrange(0, 2):  # X
                while True:
                    val = random.randrange(min_bound, max_bound + 1)
                    if val != tuple[0]:
                        break
                return (val, tuple[1])
            else:  # Y
                while True:
                    val = random.randrange(min_bound, max_bound + 1)
                    if val != tuple[0]:
                        break
                return (tuple[0], val)
