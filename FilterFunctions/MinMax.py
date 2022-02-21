def MinMaxArray(fitnesses, vectors):

    ret = []

    goal_vectors, normalization_vectors = vectors

    for nv, gv in zip(normalization_vectors, goal_vectors):
        ret.append(MinMax(fitnesses, nv, gv))

    ret = tuple(ret)
    return ret


def MinMax(fitnesses, normalization_vector, goal_vector):

    if (fitnesses[0] - goal_vector[0]) <= 0:
        ret = (goal_vector[1] - fitnesses[1]) / normalization_vector[1]
    else:
        ret = max(
            (fitnesses[0] - goal_vector[0]) / normalization_vector[0],
            (goal_vector[1] - fitnesses[1]) / normalization_vector[1],
        )

    return ret
