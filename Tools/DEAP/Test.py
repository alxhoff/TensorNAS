from deap import base, tools
import multiprocessing
from TensorNAS.Tools.Visualisation import IndividualRecord
import numpy as np

from TensorNAS.Core.Individual import TensorNASIndividual


def setup_DEAP(
    creator,
    toolbox,
    objective_weights,
    multithreaded=False,
    distributed=False,
    thread_count=0,
):
    creator.create("FitnessMulti", base.Fitness, weights=objective_weights)
    creator.create("Individual", TensorNASIndividual, fitness=creator.FitnessMulti)

    if distributed:
        from scoop import futures

        print("Running distributed using futures")
        toolbox.register("map", futures.map)
    elif multithreaded:
        if thread_count > 0:
            pool = multiprocessing.Pool(processes=thread_count)
            print("Running on 1 host using {} threads".format(thread_count))
        elif thread_count == -1:
            import os

            pool = multiprocessing.Pool(processes=os.cpu_count())
            print("Running on 1 host using {} threads".format(os.cpu_count()))
        else:
            pool = multiprocessing.Pool()
            print("Running on 1 host using no thread count limit")
        toolbox.register("map", pool.starmap)


def register_DEAP_individual_gen_func(creator, toolbox, ind_gen_func):
    # Function for creating individual (block architecture)
    toolbox.register("get_block_architecture", ind_gen_func)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.get_block_architecture,
        n=1,
    )


class DEAPTest:
    def __init__(self, pop_size, gen_count, toolbox):
        from Demos import get_global

        self.pop_size = pop_size
        self.gen_count = gen_count
        self.evaluate = None
        self.mate = None
        self.mutate = None
        self.select = None
        self.ir = IndividualRecord()
        self.history = tools.History()

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        self.pop = toolbox.population(n=self.pop_size)

        if get_global("existing_generation"):
            from TensorNAS.Tools.JSONImportExport import ImportGeneration

            exist_pop = ImportGeneration(get_global("existing_generation"))

            for i, ind in enumerate(exist_pop[: len(self.pop)]):
                self.pop[i].block_architecture = ind

        self.history.update(self.pop)
        self.hof = tools.ParetoFront(self._compare_individual)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max)

    def set_evaluate(self, toolbox, func):
        self.evaluate = func
        toolbox.register("evaluate", func)

    def set_mate(self, toolbox, func):
        self.mate = func
        toolbox.register("mate", func)
        toolbox.decorate("mate", self.history.decorator)

    def set_mutate(self, toolbox, func):
        self.mutate = func
        toolbox.register("mutate", func)
        toolbox.decorate("mutate", self.history.decorator)

    def set_select(self, toolbox, func):
        self.select = func
        toolbox.register("select", func)

    @staticmethod
    def _compare_individual(ind1, ind2):

        if ind1.fitness.values == ind2.fitness.values:
            return True

        return False

    def run(self, algorithm, args):

        if not self.evaluate:
            raise Exception("Evaluate function not set")

        if not self.mate:
            raise Exception("Mate function not set")

        if not self.mutate:
            raise Exception("Mutate function not set")

        if not self.select:
            raise Exception("Select function not set")

        self.pop, logbook = algorithm(args)

        return self.pop, logbook, self.hof
