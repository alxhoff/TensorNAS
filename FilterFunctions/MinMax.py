def MinMaxArray(fitnesses, vectors, weights):

    ret = []

    goal_vectors, normalization_vectors = vectors

    for nv, gv in zip(normalization_vectors, goal_vectors):
        ret.append(MinMax(fitnesses, nv, gv, weights))

    return tuple(ret)


def MinMax(fitnesses, normalization_vector, goal_vector, weights):

    import numpy as np

    fitnesses = np.array(fitnesses)
    normalization_vector = np.array(normalization_vector)
    goal_vector = np.array(goal_vector)
    weights = np.array(weights)

    ret = goal_vector - fitnesses
    ret = ret * weights
    ret = np.divide(ret, normalization_vector)
    ret = np.amax(ret)
    
    return ret
