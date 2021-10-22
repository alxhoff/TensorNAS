import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--folder",
    help="Absolute path to folder where interrupted test's output is stored",
    default=None,
)
parser.add_argument(
    "--gen", help="Generation from which the test should resume", type=int
)

parser.add_argument(
    "--config",
    help="Location of config file to be used, default is to use first found config file in current working directory, then parent directories",
    type=str,
    default=None,
)

args = parser.parse_args()

args.config = "DemoVisualWakeWordEASimple"

if __name__ == "__main__":

    from TensorNASDemos.DEAP import get_config
    from TensorNASDemos import (
        load_globals_from_config,
        load_tensorflow_params_from_config,
        set_test_train_data,
        get_global,
        evaluate_individual,
        mutate_individual,
    )
    from TensorNASDemos.DEAP import load_genetic_params_from_config, run_deap_test
    from TensorNASDemos.Datasets.VWW import GetData
    from TensorNAS.Core.Crossover import crossover_individuals_sp

    config = get_config(args=args)

    load_globals_from_config(config)
    load_genetic_params_from_config(config)
    load_tensorflow_params_from_config(config)

    train_generator, val_generator, input_tensor_shape = GetData()
    set_test_train_data(
        train_generator=train_generator,
        val_generator=val_generator,
        input_tensor_shape=input_tensor_shape,
        training_sample_size=get_global("training_sample_size"),
        test_sample_size=get_global("test_sample_size"),
    )

    pop, logbook, test = run_deap_test(
        evaluate_individual=evaluate_individual,
        crossover=crossover_individuals_sp,
        mutate=mutate_individual,
    )

    print("Done")
