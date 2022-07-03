def WeightedSumArray(fitnesses, vectors):

    ret = []

    goal_vector, normalization_vectors = vectors

    for nv in normalization_vectors:
        ret.append(WeightedSum(fitnesses, nv))

    return tuple(ret)


def WeightedSum(fitnesses, normalization_vector):

    import numpy as np

    return np.sum(np.divide(fitnesses, normalization_vector))
