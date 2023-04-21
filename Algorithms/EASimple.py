import time
import math


def TestEASimple(
    cxpb,
    mutpb,
    pop_size,
    gen_count,
    evaluate_individual,
    crossover_individual,
    mutate_individual,
    toolbox,
    test_name,
    verbose=False,
    filter_function=None,
    filter_function_args=None,
    save_individuals=True,
    generation_gap=1,
    generation_save=1,
    comment=None,
    multithreaded=True,
    log=None,
    start_gen=0,
):
    if log:
        from TensorNAS.Tools.Logging import Logger

        logger = Logger(test_name)
        logger.log("Starting test {}".format(test_name))

    from TensorNAS.Tools.DEAP.Test import DEAPTest

    test = DEAPTest(
        pop_size=pop_size,
        gen_count=gen_count,
        toolbox=toolbox,
    )

    test.set_evaluate(toolbox=toolbox, func=evaluate_individual)
    test.set_mate(toolbox=toolbox, func=crossover_individual)
    test.set_mutate(toolbox=toolbox, func=mutate_individual)

    from deap import tools

    test.set_select(toolbox=toolbox, func=tools.selTournamentDCD)

    pop, logbook = eaSimple(
        population=test.pop,
        toolbox=toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=gen_count,
        test_name=test_name,
        stats=test.stats,
        halloffame=test.hof,
        verbose=verbose,
        individualrecord=test.ir,
        save_individuals=save_individuals,
        filter_function=filter_function,
        filter_function_args=filter_function_args,
        logger=logger,
        generation_save_interval=generation_save,
        multithreaded=multithreaded,
        start_gen=start_gen,
    )

    test.ir.save(
        generation_gap,
        test_name=test_name,
        title=filter_function.__name__ if filter_function else "no filter func",
        comment=comment,
    )

    test.ir.goals(generation_gap, test_name=test_name)

    pareto_inds = test.ir.pareto(test_name=test_name)

    pareto_models = []
    for ind in pareto_inds:
        models = [
            i
            for i in pop
            if (
                (i.block_architecture.evaluation_values[0] == ind[0])
                and (i.block_architecture.evaluation_values[1] == ind[1])
            )
        ]
        if len(models):
            pareto_models.append(models[0])

    from TensorNAS.Tools import copy_pareto_model
    from Demos import set_global, get_global

    mutation_log_string = get_global("mutation_log_string")
    pareto_log_string = get_global("pareto_log_string")

    for i, pind in enumerate(pareto_models):
        copy_pareto_model(test_name, gen_count, pind.index, i)
        if logger:
            logger.log("####\nPareto Ind #{}".format(i))
            logger.log(
                pareto_log_string.format(
                    *reversed(pind.block_architecture.evaluation_values),
                )
            )
            logger.log(str(pind))
            logger.log("Mutations:")
            # IN GENERAL CASE: use for loop in range(#golas_number) to print all goals_diffs
            for mutation in pind.block_architecture.mutations:
                logger.log(
                    ("{} " + mutation_log_string).format(
                        mutation.mutation_function,
                        *mutation.evaluation_values_diff,
                    )
                )
            logger.log("####")

    if logger:
        logger.log("Done")
        logger.log("STOP")

    return pop, logbook, test


def eaSimple(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    test_name,
    stats=None,
    halloffame=None,
    verbose=__debug__,
    individualrecord=None,
    save_individuals=False,
    filter_function=None,
    filter_function_args=None,
    logger=None,
    generation_save_interval=1,
    multithreaded=False,
    start_gen=0,
):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.Tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.Tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`deap.Tools.Logbook` with the statistics of the
              evolution
    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.Tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.Tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::
        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring
    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.Tools.selTournament` and :func:`~deap.Tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.Tools.Logbook` of the evolution.
    .. note::
        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.
    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.
    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    from deap import tools
    from Demos import set_global, get_global
    from tqdm import tqdm
    import csv

    pop_size = len(population)
    retrain = get_global("retrain_every_generation")
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    mutation_log_string = get_global("mutation_log_string")
    evaluated_values_log_string = get_global("evaluated_values_log_string")

    if logger:
        from TensorNAS.Tools.Logging import Logger

        timing_log = Logger(test_name, subdir="Timing")
        start_time = time.time()
        cur_gen_start_time = start_time
        timing_log.log("Start time: {}".format(start_time))
        logger.log("Gen #0, population: {}".format(len(population)))

    with open(
        "Output/{}/generations.csv".format(test_name), "w", newline=""
    ) as csvfile:

        writer = csv.writer(csvfile, delimiter=" ")

        writer.writerow(["Test Name", "Population",
                        "Generations", "Cxpb", "Mutpb"])
        writer.writerow([test_name, pop_size, ngen, cxpb, mutpb])
        writer.writerow(["Gen #0"])
        #raw_pcount_row = ["Param Count"]
        #raw_acc_row = ["Accuracy"]
        raw_evaluated_values_row = get_global("raw_evaluated_values_row")
        filtered_fitness_row = ["Fitness"]

        weights = get_global("weights")

        for i, ind in enumerate(population):
            ind.index = i

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]

        print("GEN #0, evaluating {} new individuals".format(len(invalid_ind)))

        if multithreaded:
            from multiprocessing import set_start_method

            set_start_method("spawn", force=True)

            if save_individuals and generation_save_interval == 1:
                fitnesses = toolbox.map(
                    toolbox.evaluate,
                    [
                        (ind, test_name, start_gen, logger)
                        for i, ind in enumerate(invalid_ind)
                    ],
                )
            else:
                fitnesses = toolbox.map(
                    toolbox.evaluate,
                    [(ind, None, None, logger) for ind in invalid_ind],
                )
        else:
            fitnesses = []
            for i, ind in enumerate(tqdm(invalid_ind)):
                if save_individuals and generation_save_interval == 1:
                    ret = toolbox.evaluate(ind, test_name, start_gen, logger)
                else:
                    ret = toolbox.evaluate(ind, None, None, logger)
                fitnesses.append(ret)

        for count, (ind, fit) in enumerate(zip(invalid_ind, fitnesses)):
            for i in range(get_global("goals_number")):
                ind.block_architecture.evaluation_values.append(fit[-get_global(
                    "goals_number")+i])  # fit[] is iterated in reverse

            # Assign individuals an index so they can be copied in output folder structure if taken to next gen
            ind.index = count

            if filter_function:
                if filter_function_args:
                    ind.fitness.values = filter_function(
                        fit, filter_function_args, weights)
                else:
                    ind.fitness.values = filter_function(fit)
            else:
                ind.fitness.values = fit

            if fit[0] is not math.inf and fit[1] is not 0: # this needs goal generalization
                for i in range(get_global("goals_number")):
                    raw_evaluated_values_row[i].append(fit[i])
                
                filtered_fitness_row.append(ind.fitness.values[0])

            # determine which optimization param is currently the goal of the individual
            import numpy as np
            from TensorNAS.Core.BlockArchitecture import OptimizationGoal

            goal_vector = np.array(filter_function_args[0][0])
            norm_vector = np.array(filter_function_args[1][0])
            weights_vector = np.array(get_global("weights"))

            Fk = (np.array(ind.block_architecture.evaluation_values) - goal_vector)
            Fk_normalized = np.divide(Fk, norm_vector)
            Fk_normalized_and_wighted = Fk_normalized * weights_vector

            # find the index of the worse evaluated value
            worst_evaluated_value_index = np.argmax(Fk_normalized_and_wighted)

            # set the optimaization goal of the individual
            ind.block_architecture.optimization_goal = list(
                OptimizationGoal)[worst_evaluated_value_index]

            if hasattr(ind, "updates"):
                ind.updates.append(
                    tuple(ind.block_architecture.evaluation_values))
            else:
                ind.updates = [tuple(ind.block_architecture.evaluation_values)]
        
        for i in range(get_global("goals_number")):
            writer.writerow(raw_evaluated_values_row[i])
    
        writer.writerow(filtered_fitness_row)

        if logger:
            for x, ind in enumerate(population):
                logger.log(
                    ("####\nInd #{}, " + evaluated_values_log_string).format(
                        x,
                        *ind.block_architecture.evaluation_values,
                    )
                )
                logger.log("Mutations:")
                # IN GENERAL CASE: use for loop in range(#golas_number) to print all goals_diffs
                for mutation in ind.block_architecture.mutations:
                    logger.log(
                        ("{} " + mutation_log_string).format(
                            mutation.mutation_operation,
                            *mutation.mutation.evaluation_values_diff,
                        )
                    )
                logger.log("####")

        from deap.tools.emo import assignCrowdingDist

        assert len(population) == pop_size, "Initial population not of size {}".format(
            pop_size
        )

        assignCrowdingDist(population)

        if individualrecord:
            individualrecord.add_gen(population)

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        if logger:
            cur_time = time.time()
            timing_log.log(
                "Gen #0 finished in: {}".format(cur_gen_start_time - cur_time)
            )
            cur_gen_start_time = cur_time

        # Begin the generational process
        for gen in range(start_gen + 1, ngen + 1):

            #raw_pcount_row = ["Param Count"]
            #raw_acc_row = ["Accuracy"]
            raw_evaluated_values_row = get_global("raw_evaluated_values_row")
            filtered_fitness_row = ["Fitness"]
            writer.writerow(["Gen #{}".format(gen)])

            set_global(
                "self_mutation_probability",
                get_global("self_mutation_probability")
                + (gen - 1) * get_global("variable_mutation_generational_change"),
            )

            if logger:
                logger.log("Gen #{}, population: {}".format(
                    gen, len(population)))

            # Select the next generation individuals
            offspring = toolbox.select(population, pop_size)

            assert (
                len(offspring) == pop_size
            ), "Initial population not of size {}".format(pop_size)

            # Vary the pool of individuals
            from deap.algorithms import varAnd

            offspring = varAnd(offspring, toolbox, cxpb, mutpb)

            valid_ind = [ind for ind in offspring if ind.fitness.valid]

            if retrain == False:
                invalid_ind = [
                    ind for ind in offspring if not ind.fitness.valid]
            else:
                invalid_ind = offspring

            for i, ind in enumerate(invalid_ind):
                ind.index = i

            if logger:
                logger.log("{} existing individuals".format(len(valid_ind)))

            # Copy existing models to new generation
            from TensorNAS.Tools import copy_output_model

            for i, ind in enumerate(valid_ind):
                copy_output_model(test_name, gen, ind.index,
                                  len(invalid_ind) + i)
                logger.log(
                    "Copying existing model, index:{}/{}->{}/{}".format(
                        gen - 1, ind.index, gen, len(invalid_ind) + i
                    )
                )
                ind.index = len(invalid_ind) + i

            offspring[:] = invalid_ind + valid_ind

            # Evaluate the individuals with an invalid fitness
            if logger:
                logger.log("{} new individuals".format(len(invalid_ind)))

            print(
                "GEN #{}, evaluating {} new individuals".format(
                    gen, len(invalid_ind))
            )

            if multithreaded:
                from multiprocessing import set_start_method

                set_start_method("spawn", force=True)

                if save_individuals and ((gen + 1) % generation_save_interval) == 0:
                    for ind in invalid_ind:
                        index = offspring.index(ind)
                        ind.index = index
                    fitnesses = toolbox.map(
                        toolbox.evaluate,
                        [(ind, test_name, gen, logger) for ind in invalid_ind],
                    )
                else:
                    fitnesses = toolbox.map(
                        toolbox.evaluate,
                        [(ind, None, None, logger) for ind in invalid_ind],
                    )
            else:
                fitnesses = []
                for ind in tqdm(invalid_ind):
                    if save_individuals and generation_save_interval == 1:
                        fitnesses.append(toolbox.evaluate(
                            ind, test_name, gen, logger))
                    else:
                        fitnesses.append(
                            toolbox.evaluate(ind, None, None, None, logger)
                        )

            for count, (ind, fit) in enumerate(zip(invalid_ind, fitnesses)):

                if filter_function:
                    if filter_function_args:
                        ind.fitness.values = filter_function(
                            fit, filter_function_args, weights)
                    else:
                        ind.fitness.values = filter_function(fit)

                evaluation_values_diff = []
                for i in range(get_global("goals_number")):
                    ind.block_architecture.prev_evaluation_values.append(
                        ind.block_architecture.evaluation_values[i])
                    ind.block_architecture.evaluation_values[i] = fit[-get_global(
                        "goals_number")+i]

                    evaluation_values_diff.append(ind.block_architecture.evaluation_values[i]
                                                  - ind.block_architecture.prev_evaluation_values[i])

                for i in reversed(ind.block_architecture.mutations):
                    if i.pending == False:
                        break

                    i.evaluation_values_diff = evaluation_values_diff
                    i.propogate_mutation_results()

                if hasattr(ind, "updates"):
                    ind.updates.append(
                        tuple(ind.block_architecture.evaluation_values) +
                        tuple(ind.fitness.values)
                    )
                else:
                    ind.updates = [
                        (
                            tuple(ind.block_architecture.evaluation_values) +
                            tuple(ind.fitness.values)
                        )
                    ]

            assignCrowdingDist(offspring)

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            if individualrecord:
                individualrecord.add_gen(population)

            for i in population:
                if (
                    i.block_architecture.evaluation_values[0] is not math.inf # this needs goal generalization
                    and i.block_architecture.evaluation_values[1] is not 0
                ):
                    for j in range(get_global("goals_number")):
                        raw_evaluated_values_row[j].append(
                        i.block_architecture.evaluation_values[j])

                    filtered_fitness_row.append(str(i.fitness.values[0]))
            
            for i in range(get_global("goals_number")):
                        writer.writerow(raw_evaluated_values_row[i])

            writer.writerow(filtered_fitness_row)

            if logger:
                for x, ind in enumerate(population):
                    logger.log(
                        ("####\nInd #{}, " + evaluated_values_log_string).format(
                            x,
                            *ind.block_architecture.evaluation_values,
                        )
                    )
                    logger.log("Mutations:")
                    for mutation in ind.block_architecture.mutations:
                        logger.log(
                            ("{} " + mutation_log_string).format(
                                mutation.mutation_function,
                                *mutation.evaluation_values_diff,
                            )
                        )
                    logger.log("####")

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

            if logger:
                cur_time = time.time()
                timing_log.log(
                    "Gen #{} finished in: {}".format(
                        gen, cur_gen_start_time - cur_time)
                )
                cur_gen_start_time = cur_time

        if logger:
            timing_log.log("Total time: {}".format(time.time() - start_time))
            timing_log.log("STOP")

    return population, logbook
