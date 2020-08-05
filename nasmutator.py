from enum import Enum, auto
import random


class IntMutationOperators(Enum):
    STEP = auto()
    RANDOM = auto()


class TupleMutationOperators(IntMutationOperators):
    SYNC_STEP = auto()  # Both values in the tuple are mutated together
    SYNC_RANDOM = auto()


def mutate_int(val, min_bound, max_bound, operator=IntMutationOperators.STEP):
    if operator == IntMutationOperators.RANDOM:
        return random.randrange(min_bound, max_bound)
    elif operator == IntMutationOperators.STEP:
        if random.randrange(0, 1) and val < max_bound:
            return val + 1
        else:
            if val > 1:
                return val - 1
            else:
                return val + 1


def mutate_tuple(
    tuple, min_bound, max_bound, operator=TupleMutationOperators.SYNC_STEP
):
    ret = tuple
    while True:  # loop until a mutation was performed
        if operator == TupleMutationOperators.STEP:
            if random.randrange(0, 1):  # Inc
                if random.randrange(0, 1):  # X value
                    if tuple[0] < max_bound:
                        ret[0] += 1
                        return ret
                    else:
                        continue
                else:  # Y value
                    if tuple[0] > 1:
                        ret[1] += 1
                        return ret
                    else:
                        continue
            else:  # Dec
                if random.randrange(0, 1):  # X value
                    if tuple[0] < max_bound:
                        ret[0] -= 1
                        return ret
                    else:
                        continue
                else:  # Y value
                    if tuple[0] > 1:
                        ret[1] -= 1
                        return ret
                    else:
                        continue
        elif operator == TupleMutationOperators.SYNC_STEP:
            if random.randrange(0, 1):  # Inc
                if tuple[0] < max_bound and tuple[1] < max_bound:
                    ret[0] += 1
                    ret[1] += 1
                    return ret
                else:
                    continue
            else:  # Dec
                if tuple[0] > 1 and tuple[1] > 1:
                    ret[0] -= 1
                    ret[1] -= 1
                    return ret
                else:
                    continue
        elif operator == TupleMutationOperators.SYNC_RANDOM:
            while (
                True
            ):  # generate a different value to what we currently have (referenced using X val)
                val = random.randrange(0, max_bound)
                if val != tuple[0]:
                    break
            ret[0] = val
            ret[1] = val
            return ret
        elif operator == TupleMutationOperators.RANDOM:
            if random.randrange(0, 1):  # X
                while True:
                    val = random.randrange(0, max_bound)
                    if val != tuple[0]:
                        break
                ret[0] = val
            else:  # Y
                while True:
                    val = random.randrange(0, max_bound)
                    if val != tuple[0]:
                        break
                ret[1] = val
            return ret
