def MinMaxArray(fitnesses, vectors):

    ret = []

    goal_vectors, normalization_vectors, activation_vector = vectors

    for nv, gv in zip(normalization_vectors, goal_vectors):
        ret.append(MinMax(fitnesses, nv, gv, activation_vector))

    return tuple(ret), activation_vector


def MinMax(fitnesses, normalization_vector, goal_vector, activation_vector):

    # If param count goal has been reached then push accuracy only
    if activation_vector[0] == 1 :
        if (fitnesses[0] - goal_vector[0]) <= 0:
            ret = max(((goal_vector[1]*activation_vector[1]) - (fitnesses[1]*activation_vector[1])) / (normalization_vector[1]*activation_vector[1]),
                      ((fitnesses[2]*activation_vector[2]) - (goal_vector[2]*activation_vector[2])) / (normalization_vector[2]*activation_vector[1]),
                      ((fitnesses[3]*activation_vector[3]) - (goal_vector[3]*activation_vector[3])) / (normalization_vector[3]*activation_vector[3])
                      )
        else:
            ret = max(((fitnesses[0]*activation_vector[0]) - (goal_vector[0]*activation_vector[0])) / (normalization_vector[0]*activation_vector[0]),
                   ((goal_vector[1]*activation_vector[1]) - (fitnesses[1]*activation_vector[1])) / (normalization_vector[1]*activation_vector[1]),
                   ((fitnesses[2]*activation_vector[2]) - (goal_vector[2]*activation_vector[2])) / (normalization_vector[2]*activation_vector[2]),
                   ((fitnesses[3]*activation_vector[3]) - (goal_vector[3]*activation_vector[3])) / (normalization_vector[3]*activation_vector[3]))
                   
              
    else:
        ret = max(((fitnesses[0]*activation_vector[0]) - (goal_vector[0]*activation_vector[0])) / (normalization_vector[0]*activation_vector[0]),
                   ((goal_vector[1]*activation_vector[1]) - (fitnesses[1]*activation_vector[1])) / (normalization_vector[1]*activation_vector[1]),
                   ((fitnesses[2]*activation_vector[2]) - (goal_vector[2]*activation_vector[2])) / (normalization_vector[2]*activation_vector[2]),
                   ((fitnesses[3]*activation_vector[3]) - (goal_vector[3]*activation_vector[3])) / (normalization_vector[3]*activation_vector[3]))

    return ret
