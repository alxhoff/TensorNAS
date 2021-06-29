from deap import base, creator, tools
import multiprocessing
from tensornas.tools.visualization import IndividualRecord
import numpy as np

from tensornas.core.individual import Individual


class DEAPTest:
    def __init__(
        self,
        pop_size,
        gen_count,
        f_gen_individual,
        objective_weights,
        multithreaded=True,
    ):
        self.pop_size = pop_size
        self.gen_count = gen_count
        self.toolbox = base.Toolbox()
        self.evaluate = None
        self.mate = None
        self.mutate = None
        self.select = None
        self.ir = IndividualRecord()

        creator.create("FitnessMulti", base.Fitness, weights=objective_weights)
        creator.create("Individual", Individual, fitness=creator.FitnessMulti)

        if multithreaded:
            self.pool = multiprocessing.Pool()
            self.toolbox.register("map", self.pool.map)

        # Function for creating individual (block architecture)
        self.toolbox.register("get_block_architecture", f_gen_individual)
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.get_block_architecture,
            n=1,
        )

        self.history = tools.History()

        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual, n=pop_size
        )

        self.pop = self.toolbox.population(n=self.pop_size)
        self.history.update(self.pop)
        self.hof = tools.ParetoFront(self._compare_individual)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max)

    def set_evaluate(self, func):
        self.evaluate = func
        self.toolbox.register("evaluate", func)

    def set_mate(self, func):
        self.mate = func
        self.toolbox.register("mate", func)
        self.toolbox.decorate("mate", self.history.decorator)

    def set_mutate(self, func):
        self.mutate = func
        self.toolbox.register("mutate", func)
        self.toolbox.decorate("mutate", self.history.decorator)

    def set_select(self, func):
        self.select = func
        self.toolbox.register("select", func)

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
