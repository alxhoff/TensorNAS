class TensorNASIndividual:
    """
    This class is a wrapper object that allows for DEAP individuals to contain the abstract classes defined by TensorNAS Framework
    as DEAP does not allow the use of abstract classes for individuals. As such this class stores a block architecture,
    which is sub-classed from the abstract Block class, and calls into the block architecture to implement DEAP
    functionality.

    Care should be taken to stick with DEAP's requirements, especially the implementation and return values from
    the operations required for DEAP. Eg. DEAP's mutation operation requires that the newly mutated Individual
    be returned from the mutate function, it is not ok to simply mutate the individual as this will break DEAP's
    internal lists for carrying out the EA process.
    """

    def __init__(self, block_architecture):
        self.block_architecture = next(block_architecture)
        self.index = None

    def mutate(
        self,
        mutate_equally=True,
        mutation_probability=0.0,
        verbose=False,
        generation_change=0.0,
    ):
        self.index = None
        self.block_architecture.mutate(
            mutate_equally=mutate_equally,
            mutation_probability=mutation_probability,
            verbose=verbose,
            generation_change=generation_change,
        )
        return self

    def evaluate(
        self,
        train_data=None,
        train_labels=None,
        test_data=None,
        test_labels=None,
        train_generator=None,
        validation_generator=None,
        test_generator=None,
        epochs=10,
        batch_size=100,
        loss=None,
        metrics="accuracy",
        test_name=None,
        model_name=None,
        q_aware=False,
        logger=None,
        steps_per_epoch=None,
        test_steps=None,
    ):
        (param_count, accuracy,) = self.block_architecture.evaluate(
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            train_generator=train_generator,
            validation_generator=validation_generator,
            test_generator=test_generator,
            epochs=epochs,
            batch_size=batch_size,
            loss=loss,
            metrics=metrics,
            test_name=test_name,
            model_name=model_name,
            q_aware=q_aware,
            logger=logger,
            steps_per_epoch=steps_per_epoch,
            test_steps=test_steps,
        )
        return param_count, accuracy

    def print(self):
        self.block_architecture.print()

    def __str__(self):
        return str(self.block_architecture)

    def print_summary(self):
        self.block_architecture.print_summary()

    def print_tree(self):
        print(self.block_architecture.get_ascii_tree())
