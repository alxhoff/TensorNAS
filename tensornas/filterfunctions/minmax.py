def MinMaxArray(fitnesses, vectors):

    ret = []

    goal_vectors, normalization_vectors = vectors

    for nv, gv in zip(normalization_vectors, goal_vectors):
        ret.append(MinMax(fitnesses, nv, gv))

    ret = tuple(ret)
    return ret


def MinMax(fitnesses, normalization_vector, goal_vector):

    ret = max(
        (fitnesses[0] - goal_vector[0]) / normalization_vector[0],
        (goal_vector[1] - fitnesses[1]) / normalization_vector[1],
    )
    print("----")
    print(
        "Value Nr. param:"
        + str(fitnesses[0])
        + " Value Acc: "
        + str(fitnesses[1])
        + " obj:"
        + str(ret)
    )
    print(
        "Goal  Nr. param:" + str(goal_vector[0]) + " Goal  Acc: " + str(goal_vector[1])
    )
    print(
        "Norm  Nr. param:"
        + str(normalization_vector[0])
        + " Norm  Acc: "
        + str(normalization_vector[1])
    )
    print("----")
    return ret
