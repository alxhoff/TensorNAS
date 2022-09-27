from curses import raw
import time, math


def TestEASimple(
    cxpb,
    mutpb,
    pop_size,
    gen_count,
    evaluate_individual, #return param and acc and i should add the flops 
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
                (i.block_architecture.param_count == ind[0])
                and (i.block_architecture.accuracy == ind[1])
            )
        ]
        if len(models):
            pareto_models.append(models[0])

    from TensorNAS.Tools import copy_pareto_model

    for i, pind in enumerate(pareto_models):
        copy_pareto_model(test_name, gen_count, pind.index, i)
        if logger:
            logger.log("####\nPareto Ind #{}".format(i))
            logger.log(
                "Acc: {}, Param Count: {}".format(
                    pind.block_architecture.accuracy,
                    pind.block_architecture.param_count,
                )
            )
            logger.log(str(pind))
            logger.log("Mutations:")
            for mutation in pind.block_architecture.mutations:
                logger.log(
                    "{} param diff: {} acc diff: {}".format(
                        mutation.mutation_function,
                        mutation.param_diff,
                        mutation.accuracy_diff,
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

        writer.writerow(["Test Name", "Population", "Generations", "Cxpb", "Mutpb"])
        writer.writerow([test_name, pop_size, ngen, cxpb, mutpb])
        writer.writerow(["Gen #0"])
        raw_pcount_row = ["Param Count"]
        raw_acc_row = ["Accuracy"]
        raw_flops_row = ['Flops']
        raw_storage_row = ['Storage']
        filtered_fitness_row = ["Fitness"]

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
            ind.block_architecture.param_count = fit[-4]
            ind.block_architecture.accuracy = fit[-3]
            ind.block_architecture.flops = fit[-2]
            ind.block_architecture.storage = fit[-1] #i added the flops 
            # Assign individuals an index so they can be copied in output folder structure if taken to next gen
            ind.index = count
            #activation_vector = [1,1,1]

            if filter_function:
                if filter_function_args:
                    ind.fitness.values , activation_vector = filter_function(fit, filter_function_args)
                else:
                    ind.fitness.values = filter_function(fit)
            else:
                ind.fitness.values = fit

            if fit[0] is not math.inf and fit[1] is not 0:
                raw_pcount_row.append(fit[0])
                raw_acc_row.append(fit[1])
                raw_flops_row.append(fit[2])
                raw_storage_row.append(fit[3])
                filtered_fitness_row.append(ind.fitness.values[0])

            # determine which optimization param is currently the goal of the individual
            import numpy as np
            from TensorNAS.Core.BlockArchitecture import OptimizationGoal

            goal_vector = np.array(filter_function_args[0][0])
            norm_vector = np.array(filter_function_args[1][0])
            # goal vector intercepts
            m = -norm_vector[1] / norm_vector[0]
            c = goal_vector[1] - (m * goal_vector[0])

            # Intecept points
            yc = [0, c]
            xc = [-c / m, 0]

            is_above = (
                lambda p: np.cross(
                    p - np.array(yc),
                    np.array(xc) - np.array(yc),
                )
                < 0
            )
            above = is_above(
                (ind.block_architecture.param_count, ind.block_architecture.accuracy)
            )
            if above:
                if activation_vector[0] == 1:

                    if ind.block_architecture.param_count - goal_vector[0] <= 0:
                        ind.block_architecture.optimization_goal = (
                            OptimizationGoal.ACCURACY_UP
                            )
                    elif activation_vector[1] == 1:
                     ind.block_architecture.optimization_goal = (
                        OptimizationGoal.PARAMETERS_DOWN
                    )
                    elif activation_vector[2] == 1:
                        ind.block_architecture.optimization_goal = (
                            OptimizationGoal.FLOPS_DOWN )     
                    elif activation_vector[3] == 1:
                        ind.block_architecture.optimization_goal = (
                            OptimizationGoal.STORAGE_DOWN )

                elif activation_vector[1] == 1:
                     ind.block_architecture.optimization_goal = (
                        OptimizationGoal.PARAMETERS_DOWN
                    )
                elif activation_vector[2] == 1:
                    ind.block_architecture.optimization_goal = (
                        OptimizationGoal.FLOPS_DOWN )
                elif activation_vector[3] == 1:
                        ind.block_architecture.optimization_goal = (
                            OptimizationGoal.STORAGE_DOWN )

            else:
                if activation_vector[0] == 1:
                    ind.block_architecture.optimization_goal = OptimizationGoal.ACCURACY_UP

            if hasattr(ind, "updates"):
                ind.updates.append(
                    (
                        ind.block_architecture.param_count,
                        ind.block_architecture.accuracy,
                        ind.block_architecture.flops,
                        ind.block_architecture.storage

                    )
                )
            else:
                ind.updates = [
                    (
                        ind.block_architecture.param_count,
                        ind.block_architecture.accuracy,
                        ind.block_architecture.flops,
                        ind.block_architecture.storage
                    )
                ]

        writer.writerow(raw_pcount_row)
        writer.writerow(raw_acc_row)
        writer.writerow(raw_flops_row)
        writer.writerow(raw_storage_row)
        writer.writerow(filtered_fitness_row)

        if logger:
            for x, ind in enumerate(population):
                logger.log(
                    "####\nInd #{}, params:{}, acc:{}%, flops:{}, storage:{}".format(
                        x,
                        ind.block_architecture.param_count,
                        ind.block_architecture.accuracy,
                        ind.block_architecture.flops,
                        ind.block_architecture.storage
                    )
                )
                logger.log("Mutations:")
                for mutation in ind.block_architecture.mutations:
                    logger.log(
                        "{} param diff: {} acc diff: {}, flops diff{}".format(
                            mutation.mutation_operation,
                            mutation.param_diff,
                            mutation.accuracy_diff,
                            mutation.flops_diff,
                            mutation.storage_diff
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

            raw_pcount_row = ["Param Count"]
            raw_acc_row = ["Accuracy"]
            raw_flops_row = ['Flops']
            raw_storage_row = ['Storage']
            filtered_fitness_row = ["Fitness"]
            writer.writerow(["Gen #{}".format(gen)])

            set_global(
                "self_mutation_probability",
                get_global("self_mutation_probability")
                + (gen - 1) * get_global("variable_mutation_generational_change"),
            )

            if logger:
                logger.log("Gen #{}, population: {}".format(gen, len(population)))

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
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            else:
                invalid_ind = offspring

            for i, ind in enumerate(invalid_ind):
                ind.index = i

            if logger:
                logger.log("{} existing individuals".format(len(valid_ind)))

            # Copy existing models to new generation
            from TensorNAS.Tools import copy_output_model

            for i, ind in enumerate(valid_ind):
                copy_output_model(test_name, gen, ind.index, len(invalid_ind) + i)
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
                "GEN #{}, evaluating {} new individuals".format(gen, len(invalid_ind))
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
                        fitnesses.append(toolbox.evaluate(ind, test_name, gen, logger))
                    else:
                        fitnesses.append(
                            toolbox.evaluate(ind, None, None, None, logger)
                        )

            for count, (ind, fit) in enumerate(zip(invalid_ind, fitnesses)):

                if filter_function:
                    if filter_function_args:
                        ind.fitness.values, temp_activation_vec = filter_function(fit, filter_function_args)
                    else:
                        ind.fitness.values = filter_function(fit)

                ind.block_architecture.prev_param_count = (
                    ind.block_architecture.param_count
                )
                ind.block_architecture.param_count = fit[-4]
                ind.block_architecture.prev_accuracy = ind.block_architecture.accuracy
                ind.block_architecture.accuracy = fit[-3]
                ind.block_architecture.prev_flops = ind.block_architecture.flops
                ind.block_architecture.flops = fit[-2]
                ind.block_architecture.prev_storage = ind.block_architecture.storage
                ind.block_architecture.storage = fit[-1]

                acc_diff = (
                    ind.block_architecture.accuracy
                    - ind.block_architecture.prev_accuracy
                )
                param_count_diff = (
                    ind.block_architecture.param_count
                    - ind.block_architecture.prev_param_count
                )

                flop_diff = (ind.block_architecture.flops - ind.block_architecture.prev_flops)
                storage_diff = (ind.block_architecture.storage - ind.block_architecture.prev_storage)

                for i in reversed(ind.block_architecture.mutations):
                    if i.pending == False:
                        break

                    i.accuracy_diff = acc_diff
                    i.param_diff = param_count_diff
                    i.flops_diff = flop_diff
                    i.storage_diff = storage_diff
                    i.propogate_mutation_results()

                if hasattr(ind, "updates"):
                    ind.updates.append(
                        (
                            ind.block_architecture.param_count,
                            ind.block_architecture.accuracy,
                            ind.block_architecture.flops,
                            ind.block_architecture.storage,
                            ind.fitness.values,
                        )
                    )
                else:
                    ind.updates = [
                        (
                            ind.block_architecture.param_count,
                            ind.block_architecture.accuracy,
                            ind.block_architecture.flops,
                            ind.block_architecture.storage,
                            ind.fitness.values,
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
                    i.block_architecture.param_count is not math.inf
                    and i.block_architecture.accuracy is not 0
                ):
                    raw_pcount_row.append(i.block_architecture.param_count)
                    raw_acc_row.append(i.block_architecture.accuracy)
                    raw_flops_row.append(i.block_architecture.flops)
                    raw_storage_row.append(i.block_architecture.storage)
                    filtered_fitness_row.append(str(i.fitness.values[0]))

            writer.writerow(raw_pcount_row)
            writer.writerow(raw_acc_row)
            writer.writerow(raw_flops_row)
            writer.writerow(raw_storage_row)
            writer.writerow(filtered_fitness_row)

            if logger:
                for x, ind in enumerate(population):
                    logger.log(
                        "####\nInd #{}, params:{}, acc:{}%, flops:{}, storage:{}".format(
                            x,
                            ind.block_architecture.param_count,
                            ind.block_architecture.accuracy,
                            ind.block_architecture.flops,
                            ind.block_architecture.storage
                        )
                    )
                    logger.log("Mutations:")
                    for mutation in ind.block_architecture.mutations:
                        logger.log(
                            "{} param diff: {} acc diff: {}, flops diff:{}, storage:{}".format(
                                mutation.mutation_function,
                                mutation.param_diff,
                                mutation.accuracy_diff,
                                mutation.flops_diff,
                                mutation.storage_diff

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
                    "Gen #{} finished in: {}".format(gen, cur_gen_start_time - cur_time)
                )
                cur_gen_start_time = cur_time

        if logger:
            timing_log.log("Total time: {}".format(time.time() - start_time))
            timing_log.log("STOP")

    return population, logbook
