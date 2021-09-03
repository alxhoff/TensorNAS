def StepFilter(fitness):

    if fitness[1] > 50:
        return fitness
    else:
        return fitness[0], 0
