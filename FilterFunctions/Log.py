def LogFilter(fitnesses):
    """Tales a tuple fitness of param count and accuracy, must return the same."""
    from math import log

    acc = fitnesses[1]
    param_count = fitnesses[0] * log(acc + 1, 2)
    return acc, param_count
